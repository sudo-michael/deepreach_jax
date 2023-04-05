import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import argparse

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct

import sys

# TODO: make this into a package with setuptools so we don't have to do this
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import SirenNet
from dataio import create_dataset_sampler, xy_grid
from hj_functions import initialize_hji_loss, initialize_opt_ctrl_fn
from utils import unnormalize_value_function, normalize_value_function

import wandb
from train import train

import orbax.checkpoint as orbax
from flax.training import checkpoints


@struct.dataclass
class DatasetState:
    counter: int
    pretrain_end: int  # for only training ground truth value function at t=t_min
    counter_end: int  # when time can finally be sampled at t_max
    batch_size: int
    samples_at_t_min: int
    t_min: float
    t_max: float
    # dynamics
    velocity: float
    omega_max: float
    # initial value function
    collision_r: float
    # normalize state space into range [-1, 1]
    alpha: dict
    beta: dict
    # value function normaliziation
    norm_to: float = 0.02
    mean: float = 0.25
    var: float = 0.5


def main(args):
    # normalization from world space to [-1, 1]^d
    alpha = {
        "x": 2.0,
        "y": 2.0,
        "theta": args.angle_alpha * np.pi,
    }
    beta = {
        "x": 0.0,
        "y": 0.0,
        "theta": 0.0,
    }

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    layers = [args.num_nl for _ in range(args.num_hl)]

    if args.periodic_transform:

        @jax.vmap
        def periodic_transform(normalized_tcoords):
            # x should be in the range of [-1, 1]^d
            # strech periodic dim to so that sin(x * alpha + beta) \in [-1, 1]
            periodic_dim_scale = alpha["theta"] + beta["theta"]
            return jnp.array(
                [
                    normalized_tcoords[0],  # t
                    normalized_tcoords[1],  # x
                    normalized_tcoords[2],  # y
                    jnp.cos(normalized_tcoords[3] * periodic_dim_scale),  # cos(theta)
                    jnp.sin(normalized_tcoords[3] * periodic_dim_scale),  # sin(theta)
                ]
            )

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
        t_min=args.t_min,
        t_max=args.t_max,
        velocity=args.velocity,
        omega_max=args.omega_max,
        collision_r=args.collision_r,
        alpha=alpha,
        beta=beta,
    )

    ckpt = {"model": state, "config": vars(args), "dataset": dataset_state}

    def unnormalize_tcoords(normalized_tcoord):
        # [-1, 1]^d to global frame
        alpha = jnp.array(
            [
                1,
                dataset_state.alpha["x"],
                dataset_state.alpha["y"],
                dataset_state.alpha["theta"],
            ]
        )
        beta = jnp.array(
            [
                0,
                dataset_state.beta["x"],
                dataset_state.beta["y"],
                dataset_state.beta["theta"],
            ]
        )
        return normalized_tcoord * alpha + beta

    def normalize_tcoords(unnormalized_tcoord):
        # global frame to [-1, 1]^d
        alpha = jnp.array(
            [
                1,
                dataset_state.alpha["x"],
                dataset_state.alpha["y"],
                dataset_state.alpha["theta"],
            ]
        )
        beta = jnp.array(
            [
                0,
                dataset_state.beta["x"],
                dataset_state.beta["y"],
                dataset_state.beta["theta"],
            ]
        )
        return (unnormalized_tcoord - beta) / alpha

    @jax.vmap
    def initial_value_function(normalized_tcoords):
        tcoords = unnormalize_tcoords(normalized_tcoords)
        center = jnp.array([1.0, 1.0])
        lx = jnp.linalg.norm(tcoords[1:3] - center) - dataset_state.collision_r
        lx = normalize_value_function(
            lx, dataset_state.norm_to, dataset_state.mean, dataset_state.var
        )
        return lx

    def scale_costates(dVdx):
        alpha = jnp.array(
            [
                dataset_state.alpha["x"],
                dataset_state.alpha["y"],
                dataset_state.alpha["theta"],
            ]
        )
        return dVdx / alpha

    @jax.vmap
    def compute_hamiltonian(nablaV, normalized_tcoords):
        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a
        tcoords = unnormalize_tcoords(normalized_tcoords)  # (t x y theta)

        dVdx = nablaV[1:]  # (x y theta)

        # TODO understand why we scale dVdx
        dVdx = scale_costates(dVdx)

        ham = dataset_state.omega_max * jnp.abs(
            dVdx[0] * tcoords[2] - dVdx[1] * tcoords[1] - dVdx[2]
        )  # Control component

        ham = ham - dataset_state.omega_max * jnp.abs(dVdx[2])  # Disturbance component
        ham = (
            ham
            + (dataset_state.velocity * (jnp.cos(tcoords[3]) - 1.0) * dVdx[0])
            + (dataset_state.velocity * jnp.sin(tcoords[3]) * dVdx[1])
        )  # Constant component
        return ham

    @jax.vmap
    def compute_opt_ctrl(nablaV):
        dVdx = nablaV[1:]  # (x y theta)
        dVdtheta = dVdx[2]

        # computes control to maximize V
        # > avoid obstacle
        opt_ctrl = dataset_state.omega_max * (
            dVdtheta > 0
        ) + -dataset_state.omega_max * (dVdtheta <= 0)
        return opt_ctrl

    dubins_3d_dataset_sampler = create_dataset_sampler(
        initial_value_function, args.num_states
    )

    # Define the loss
    loss_fn = initialize_hji_loss(state, args.min_with, compute_hamiltonian)

    def val_fn(state, epoch):
        times = [0.0, 0.5 * (args.t_max - 0.1), (args.t_max - 0.1)]
        slices = [
            {"theta": -np.pi},
            {"theta": -0.5 * np.pi},
            {"theta": -0.0},
            {"theta": 0.5 * np.pi},
            {"theta": np.pi},
        ]

        # Create a figure
        # fig = plt.figure(figsize=(5 * num_times, 5 * num_thetas))
        fig, ax = plt.subplots(
            nrows=len(times),
            ncols=len(slices),
            sharex=True,
            sharey=True,
            figsize=(5 * len(slices), 5 * len(times)),
            # constrained_layout=True
        )

        # Get the meshgrid in the (x, y) coordinate
        grid_points = 200
        mgrid_coords = xy_grid(200, x_max=dataset_state.alpha['x'], y_max=dataset_state.alpha['y'])

        for time, row in zip(times, ax):
            for slice, col in zip(slices, row):
                time_coords = np.ones((mgrid_coords.shape[0], 1)) * time
                theta_coords = np.ones((mgrid_coords.shape[0], 1)) * slice["theta"]
                unnormalized_tcoords = np.concatenate(
                    (time_coords, mgrid_coords, theta_coords), axis=1
                )
                V = state.apply_fn(
                    state.params, jnp.array(normalize_tcoords(unnormalized_tcoords))
                )

                V = np.array(V)
                V = V.reshape((grid_points, grid_points))

                V = unnormalize_value_function(
                    V, dataset_state.norm_to, dataset_state.mean, dataset_state.var
                )

                # Plot the zero level sets
                V = (V <= 0.001) * 1.0

                # Plot the actual data
                im = col.imshow(
                    V.T,
                    cmap="bwr",
                    origin="lower",
                    extent=(-1.0, 1.0, -1.0, 1.0),
                )
                col.set_title(f"t={time:0.2f}, theta={slice['theta']:0.2f}")
                # fig.colorbar(s)

        fig.colorbar(im, ax=ax.ravel().tolist())
        wandb.log({"brt": fig}, step=epoch)
        plt.close(fig)

    if args.validate:
        ckpt_dir = os.path.join(args.logging_root, args.experiment_name, "checkpoints")
        ckpt = checkpoints.restore_checkpoint(ckpt_dir, target=ckpt)
        dataset_state = ckpt["dataset"]
        state = ckpt["model"]
        val_fn(state, 0)

        # opt_ctrl_fn = initialize_opt_ctrl_fn(state, compute_opt_ctrl)
        # tstate = jnp.array([1.0, 0.8, 0.8, -0.5 * jnp.pi]).reshape(1, -1)
        # V = state.apply_fn(state.params, normalize_tcoords(tstate))
        # opt_ctrl_fn(state.params, normalize_tcoords(tstate))

        # def controller(t, tstate, eps=0.1):
        #     V = unormalize_value_function(
        #         state.apply_fn(state.params, normalize_tcoords(tstate)),
        #         dataset_state.norm_to,
        #         dataset_state.mean,
        #         dataset_state.var,
        #     ).item()
        #     if V < eps:
        #         return opt_ctrl_fn(state.params, normalize_tcoords(tstate)).item()
        #     else:
        #         if t % 2 == 0:
        #             return dataset_state.omega_max
        #         else:
        #             return -dataset_state.omega_max

        # def simulate(tstate, u, dt=0.05):
        #     t = tstate[0][0]
        #     theta = tstate[0][3]
        #     x_dot = jnp.array([0, jnp.cos(theta), jnp.sin(theta), u])

        #     return tstate + x_dot * 0.05
        # X = []
        # Y = []
        # for t in range(100):
        #     u = controller(t, tstate)
        #     next_tstate = simulate(tstate, u)
        #     tstate = next_tstate

        #     X.append(tstate[0][1])
        #     Y.append(tstate[0][2])
        # plt.scatter(X, Y)
        # plt.show()
    else:
        train(
            key,
            state,
            dataset_state,
            loss_fn,
            val_fn,
            dubins_3d_dataset_sampler,
            ckpt,
            args.epochs,
            args.epochs_till_checkpoint,
            args.logging_root,
            args.experiment_name,
        )


if __name__ in "__main__":

    def get_args():
        p = argparse.ArgumentParser()
        # fmt: off
        p.add_argument("--experiment-name", type=str, required=True)
        p.add_argument("--wandb", action='store_true')
        p.add_argument("--logging-root", type=str, default='logs')
        p.add_argument('--epochs', type=int, default=100_000)
        p.add_argument('--epochs-till-checkpoint', type=int, default=2_000)
        p.add_argument('--pretrain-end', type=int, default=2_000)
        p.add_argument('--counter-end', type=int, default=110_000)
        p.add_argument('--batch_size', type=int, default=65_000)
        p.add_argument('--samples-at-t-min', type=int, default=10_000)
        p.add_argument('--lr', type=float, default=2e-5)
        p.add_argument('--seed', type=int, default=1)
        p.add_argument("--validate", action='store_true')
        # siren
        p.add_argument("--num-hl", type=int, default=2, required=False, 
                       help="The number of hidden layers"
        )
        p.add_argument("--num-nl", type=int, default=512, required=False, 
                       help="Number of neurons per hidden layer.",
        )
        p.add_argument("--periodic-transform", action='store_true',
                       help="convert theta to cos(theta) sin(theta)",
        )
        # brt
        p.add_argument("--min-with", type=str, default="target", required=False, choices=["none", "zero", "target"], 
                       help="BRS vs BRT computation",
        )
        p.add_argument("--t-min", type=float, default=0.0, required=False, 
                       help="Start time of the simulation",
        )
        p.add_argument("--t-max", type=float, default=1.0, required=False, 
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
        p.add_argument("--velocity", type=float, default=0.6, required=False, 
                       help="Speed of the dubins car",
        )
        p.add_argument("--omega-max", type=float, default=1.1, required=False, 
                       help="Turn rate of the car") 
        # fmt: on
        args = p.parse_args()

        assert args.pretrain_end < args.epochs

        return args

    args = get_args()

    if args.wandb:
        wandb.init(
            project="deepreach_jax",
            entity="mlu",
            config=vars(args),
            save_code=True,
            name=args.experiment_name,
        )
    main(args)
