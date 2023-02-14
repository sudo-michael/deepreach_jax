import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import math
import argparse

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from modules import SirenNet
import optax
import orbax.checkpoint as orbax
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints

from dataio import DatasetState, create_dubins_3d_dataset_sampler, xy_grid
from loss_functions import initialize_hji_air3D

import tqdm

import wandb


def main(args):

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    layers = [args.num_nl for _ in range(args.num_hl)]

    if args.periodic_transform:

        @jax.vmap
        def periodic_transform(x):
            # x should be in the range of [-1, 1]^d
            # strech periodic dim to so that sin(x * alpha + beta) \in [-1, 1]
            periodic_dim_scale = args.angle_alpha * jnp.pi
            return jnp.array([x[0], x[1], jnp.cos(x[2] * periodic_dim_scale), jnp.sin(x[2] * periodic_dim_scale)])

        model = SirenNet(hidden_layers=layers, transform_fn=periodic_transform)
        num_states = periodic_transform(jnp.ones((1, args.num_states))).shape[-1]
    else:
        model = SirenNet(hidden_layers=layers)
        num_states = args.num_states

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(
            model_key,
            jnp.ones((1, num_states)),
        ),
        tx=optax.adam(learning_rate=args.lr),
    )

    dataset_state = DatasetState(
        counter=0,
        pretrain_end=args.pretrain_end,
        counter_end=args.counter_end,
        batch_size=args.batch_size,
        samples_at_t_min=args.samples_at_t_min,
        velocity=args.velocity,
        omega_max=args.omega_max,
        angle_alpha=args.angle_alpha * np.pi,
    )

    dubins_3d_dataset_sampler = create_dubins_3d_dataset_sampler(
        t_min=args.t_min, t_max=args.t_max, collision_r=args.collision_r
    )

    key, dataset_key = jax.random.split(key)
    dataset_state, tcoords, gt = dubins_3d_dataset_sampler(dataset_key, dataset_state)

    ckpt = {"model": state, "config": vars(args)}

    # Define the loss
    loss_fn = initialize_hji_air3D(state, dataset_state, args.min_with)

    @jax.jit
    def update(state, tcoords, source_boundary_values, dirichlet_mask):
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, tcoords, source_boundary_values, dirichlet_mask
        )
        state = state.apply_gradients(grads=grads)
        return state, loss

    root_path = os.path.join(args.logging_root, args.experiment_name)

    def val_fn(state, ckpt_dir, epoch):
        # Time values at which the function needs to be plotted
        times = [0.0, 0.5 * (args.t_max - 0.1), (args.t_max - 0.1)]
        num_times = len(times)

        # Theta slices to be plotted
        thetas = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
        num_thetas = len(thetas)

        # Create a figure
        fig = plt.figure(figsize=(5 * num_times, 5 * num_thetas))

        # Get the meshgrid in the (x, y) coordinate
        grid_points = 200
        mgrid_coords = xy_grid(200)

        # Start plotting the results
        for i in range(num_times):
            time_coords = np.ones((mgrid_coords.shape[0], 1)) * times[i]

            for j in range(num_thetas):
                theta_coords = np.ones((mgrid_coords.shape[0], 1)) * thetas[j]
                theta_coords = theta_coords / (args.angle_alpha * np.pi)
                coords = np.concatenate(
                    (time_coords, mgrid_coords, theta_coords), axis=1
                )
                model_out = state.apply_fn(state.params, jnp.array(coords))

                model_out = np.array(model_out)
                model_out = model_out.reshape((grid_points, grid_points))

                # Unnormalize the value function
                norm_to = 0.02
                mean = 0.25
                var = 0.5
                model_out = (model_out * var / norm_to) + mean

                # Plot the zero level sets
                model_out = (model_out <= 0.001) * 1.0

                # Plot the actual data
                ax = fig.add_subplot(num_times, num_thetas, (j + 1) + i * num_thetas)
                ax.set_title("t = %0.2f, theta = %0.2f" % (times[i], thetas[j]))
                s = ax.imshow(
                    model_out.T,
                    cmap="bwr",
                    origin="lower",
                    extent=(-1.0, 1.0, -1.0, 1.0),
                )
                fig.colorbar(s)

        wandb.log({"brt": fig}, step=epoch)
        fig.clear()

    root_path = os.path.join(args.logging_root, args.experiment_name)
    model_dir = root_path
    import utils
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    train_losses = []
    for epoch in tqdm.tqdm(range(args.epochs)):

        key, dataset_key = jax.random.split(key)
        dataset_state, tcoords, gt = dubins_3d_dataset_sampler(
            dataset_key, dataset_state
        )
        state, train_loss = update(
            state, tcoords, gt["source_boundary_values"], gt["dirichlet_mask"]
        )
        train_losses.append(train_loss.item())

        if not epoch % args.epochs_till_checkpoint and epoch:
            orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
            checkpoints.save_checkpoint(ckpt_dir=checkpoints_dir,
                                        target=ckpt,
                                        step=epoch,
                                        overwrite=False,
                                        keep=2,
                                        orbax_checkpointer=orbax_checkpointer)
            wandb.log({"train_loss": train_losses[-1]}, step=epoch)
            print(f"Epoch: {epoch} train_loss: {train_losses[-1]}")

            val_fn(state, checkpoints_dir, epoch)


if __name__ in "__main__":

    def get_args():
        p = argparse.ArgumentParser()
        # fmt: off
        p.add_argument("--experiment-name", type=str, required=True)
        p.add_argument("--wandb", action='store_true')
        p.add_argument("--logging-root", type=str, default='logs')
        p.add_argument('--epochs', type=int, default=120_000)
        p.add_argument('--epochs-till-checkpoint', type=int, default=2_000)
        p.add_argument('--pretrain-end', type=int, default=2_000)
        p.add_argument('--counter-end', type=int, default=110_000)
        p.add_argument('--batch_size', type=int, default=65_000)
        p.add_argument('--samples-at-t-min', type=int, default=10_000)
        p.add_argument('--lr', type=float, default=2e-5)
        p.add_argument('--seed', type=int, default=1)

        # siren
        p.add_argument("--num-hl", type=int, default=2, required=False, 
                       help="The number of hidden layers"
        )
        p.add_argument("--num-nl", type=int, default=512, required=False, 
                       help="Number of neurons per hidden layer.",
        )
        p.add_argument("--periodic-transform", type=bool, default=False,
                       help="convert theta to cos(theta) sin(theta)",
        )
        # brt
        p.add_argument("--min-with", type=str, default="target", required=False, choices=["none", "zero", "target"], 
                       help="BRS vs BRT computation",
        )
        p.add_argument("--t-min", type=float, default=0.0, required=False, 
                       help="Start time of the simulation",
        )
        p.add_argument("--t-max", type=float, default=1.1, required=False, 
                       help="End time of the simulation"
        )
        # initial value function
        p.add_argument("--collision-r", type=float, default=0.25, required=False, 
                       help="Collision radius between vehicles",
        )
        # dynamics
        p.add_argument("--num-states", type=int, default=4, required=False, 
                       help="Number of states in system including time",
        )
        p.add_argument("--angle-alpha", type=float, default=1.2, required=False, 
                       help="Angle alpha coefficient.",
        )
        p.add_argument("--velocity", type=float, default=0.75, required=False, 
                       help="Speed of the dubins car",
        )
        p.add_argument("--omega-max", type=float, default=3.0, required=False, 
                       help="Turn rate of the car") 
        # fmt: on
        args = p.parse_args()

        assert args.pretrain_end < args.epochs

        return args

    args = get_args()

    if args.wandb:
        wandb.init(project="deepreach_jax", entity="mlu", config=vars(args), save_code=True)
    main(args)
