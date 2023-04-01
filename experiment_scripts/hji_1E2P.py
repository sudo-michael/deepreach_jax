import matplotlib

matplotlib.use("Agg")
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
# TODO: make this into a package with setuptools
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from modules import SirenNet
from dataio import create_dataset_sampler, xy_grid
from loss_functions import initialize_hji_loss
from utils import unormalize_value_function, normalize_value_function, cond_mkdir

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
    velocity_evader: float
    velocity_persuer: float
    omega_evader: float
    omega_persuer: float
    # initial value function
    collision_r: float
    # normalize state space into range [-1, 1]
    alpha: dict
    beta: dict
    # value function normaliziation
    norm_to: float = 0.02
    mean: float = 0.25
    var: float = 0.5

def create_normalization_fn(dataset_state):
    alpha = jnp.array(
        [
            1,
            dataset_state.alpha["x"],
            dataset_state.alpha["y"],
            dataset_state.alpha["x"],
            dataset_state.alpha["y"],
            dataset_state.alpha["x"],
            dataset_state.alpha["y"],
            dataset_state.alpha["theta"],
            dataset_state.alpha["theta"],
            dataset_state.alpha["theta"],
        ]
    )

    beta = jnp.array(
        [
            0,
            dataset_state.beta["x"],
            dataset_state.beta["y"],
            dataset_state.beta["x"],
            dataset_state.beta["y"],
            dataset_state.beta["x"],
            dataset_state.beta["y"],
            dataset_state.beta["theta"],
            dataset_state.beta["theta"],
            dataset_state.beta["theta"],
        ]
    )

    def unnormalize_tcoords(normalized_tcoord):
        # [-1, 1]^d to global frame
        return normalized_tcoord * alpha + beta

    def normalize_tcoords(unnormalized_tcoord):
        # global frame to [-1, 1]^d
        return (unnormalized_tcoord - beta) / alpha

    def scale_costates(dVdx):
        return dVdx / alpha[1:]
    
    return unnormalize_tcoords, normalize_tcoords, scale_costates

def create_hj_fn(dataset_state):
    unnormalize_tcoords, normalize_tcoords, scale_costates = create_normalization_fn(dataset_state)

    @jax.vmap
    def initial_value_function(normalized_tcoords):
        # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
        # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].

        tcoords = unnormalize_tcoords(normalized_tcoords)
        lx1 = jnp.linalg.norm(tcoords[1:3] - tcoords[3:5]) - dataset_state.collision_r
        lx2 = jnp.linalg.norm(tcoords[1:3] - tcoords[5:7]) - dataset_state.collision_r
        lx = jnp.minimum(lx1, lx2)
        lx = normalize_value_function(
            lx, dataset_state.norm_to, dataset_state.mean, dataset_state.var
        )
        return lx

    @jax.vmap
    def compute_hamiltonian(nablaV, normalized_tcoords):
        tcoords = unnormalize_tcoords(normalized_tcoords)
        # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
        # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].

        dVdx = nablaV[1:]

        # TODO understand why we scale dVdx
        dVdx = scale_costates(dVdx)

        dVdx_e, dVdy_e, dVdtheta_e = dVdx[0], dVdx[1], dVdx[6]
        dVdx_p1, dVdy_p1, dVdtheta_p1 = dVdx[2], dVdx[3], dVdx[7]
        dVdx_p2, dVdy_p2, dVdtheta_p2 = dVdx[4], dVdx[5], dVdx[8]

        x_e, y_e, theta_e = tcoords[1], tcoords[2], tcoords[7]
        x_p1, y_p1, theta_p1 = tcoords[3], tcoords[4], tcoords[8]
        x_p2, y_p2, theta_p2 = tcoords[5], tcoords[6], tcoords[9]

        ham_evader = dataset_state.velocity_evader * dVdx_e * jnp.cos(
            theta_e
        ) + dataset_state.velocity_evader * dVdy_e * jnp.sin(theta_e)
        ham_evader = ham_evader + dataset_state.omega_evader * jnp.abs(dVdtheta_e)

        ham_persuer1 = dataset_state.velocity_persuer * dVdx_p1 * jnp.cos(
            theta_p1
        ) + dataset_state.velocity_persuer * dVdy_p1 * jnp.sin(theta_p1)
        ham_persuer1 = ham_persuer1 - dataset_state.omega_persuer * jnp.abs(dVdtheta_p1)

        ham_persuer2 = dataset_state.velocity_persuer * dVdx_p2 * jnp.cos(
            theta_p2
        ) + dataset_state.velocity_persuer * dVdy_p2 * jnp.sin(theta_p2)
        ham_persuer2 = ham_persuer2 - dataset_state.omega_persuer * jnp.abs(dVdtheta_p2)

        ham = ham_evader + ham_persuer1 + ham_persuer2

        return ham

    @jax.vmap
    def compute_opt_ctrl_dstb_fn(nablaV):
        # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
        # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].
        dVdx = nablaV[1:]  
        dVdtheta_e = dVdx[6]
        dVdtheta_p1 = dVdx[7]
        dVdtheta_p2 = dVdx[8]

        # opt_ctrl to maximize V
        opt_ctrl = dataset_state.omega_evader * (dVdtheta_e > 0) + -dataset_state.omega_evader * (dVdtheta_e <= 0)

        # opt_dstb to maximize V
        opt_dstb_p1 = -dataset_state.omega_persuer * (dVdtheta_p1 > 0) + dataset_state.omega_persuer * (dVdtheta_p1 <= 0)
        opt_dstb_p2 = -dataset_state.omega_persuer * (dVdtheta_p2 > 0) + dataset_state.omega_persuer * (dVdtheta_p2 <= 0)

        return (opt_ctrl,), (opt_dstb_p1, opt_dstb_p2)
    return initial_value_function, compute_hamiltonian, compute_opt_ctrl_dstb_fn
    

def main(args):
    # normalization from world space to [-1, 1]^d
    alpha = {
        "x": 4.0,
        "y": 4.0,
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
                    normalized_tcoords[1],  # x_e
                    normalized_tcoords[2],  # y_e
                    normalized_tcoords[3],  # x_p1
                    normalized_tcoords[4],  # y_p1
                    normalized_tcoords[5],  # x_p2
                    normalized_tcoords[6],  # y_p2
                    jnp.cos(normalized_tcoords[7] * periodic_dim_scale),  # cos(theta_e)
                    jnp.sin(normalized_tcoords[7] * periodic_dim_scale),  # sin(theta_e)
                    jnp.cos(
                        normalized_tcoords[8] * periodic_dim_scale
                    ),  # cos(theta_p1)
                    jnp.sin(
                        normalized_tcoords[8] * periodic_dim_scale
                    ),  # sin(theta_p1)
                    jnp.cos(
                        normalized_tcoords[9] * periodic_dim_scale
                    ),  # cos(theta_p2)
                    jnp.sin(
                        normalized_tcoords[9] * periodic_dim_scale
                    ),  # sin(theta_p2)
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
        velocity_evader=args.velocity_e,
        velocity_persuer=args.velocity_p,
        omega_evader=args.omega_e,
        omega_persuer=args.omega_p,
        collision_r=args.collision_r,
        alpha=alpha,
        beta=beta,
    )

    ckpt = {"model": state, "config": vars(args), "dataset": dataset_state}

    initial_value_function, compute_hamiltonian, compute_opt_ctrl_dstb_fn = create_hj_fn(dataset_state)


    dubins_3d_dataset_sampler = create_dataset_sampler(
        initial_value_function, args.num_states
    )

    # Define the loss
    loss_fn = initialize_hji_loss(state, args.min_with, compute_hamiltonian)

    def val_fn(state, epoch):
        unnormalize_tcoords, normalize_tcoords, scale_costates = create_normalization_fn(dataset_state)
        times = [0.0, 0.5 * (args.t_max - 0.1), (args.t_max - 0.1)]
        x_p1s = [-1, -1, -1, -1]
        y_p1s = [0, 0, 0, 0]
        x_p2s = [1, 1, 1, 1]
        y_p2s = [0, 0, 0, 0]
        theta_es = [0, np.pi / 2, -np.pi, -np.pi / 2]
        theta_p1s = [0, 0, 0, 0]
        theta_p2s = [-np.pi, -np.pi, -np.pi, -np.pi]
        keys = ["x_p1", "y_p1", "x_p2", "y_p2", "theta_e", "theta_p1", "theta_p2"]
        slices = [
            dict(zip(keys, items))
            for items in zip(x_p1s, y_p1s, x_p2s, y_p2s, theta_es, theta_p1s, theta_p2s)
        ]

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
                ones = np.ones((mgrid_coords.shape[0], 1))
                time_coords = ones * time
                theta_evader = ones * slice["theta_e"]
                x_p1 = ones * slice["x_p1"]
                y_p1 = ones * slice["y_p1"]
                x_p2 = ones * slice["x_p2"]
                y_p2 = ones * slice["y_p2"]
                theta_evader = ones * slice["theta_e"]
                theta_p1 = ones * slice["theta_p1"]
                theta_p2 = ones * slice["theta_p2"]
                unnormalized_tcoords = np.concatenate(
                    (
                        time_coords,
                        mgrid_coords,
                        x_p1,
                        y_p1,
                        x_p2,
                        y_p2,
                        theta_evader,
                        theta_p1,
                        theta_p2,
                    ),
                    axis=1,
                )
                V = state.apply_fn(
                    state.params, jnp.array(normalize_tcoords(unnormalized_tcoords))
                )

                V = np.array(V)
                V = V.reshape((grid_points, grid_points))

                V = unormalize_value_function(
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
                col.set_title(f"t={time:0.2f}, te={slice['theta_e']:0.2f}, tp1={slice['theta_p1']:0.2f} tp2={slice['theta_p2']:0.2f}")
                # fig.colorbar(s)

        fig.suptitle('x=0, y=0, xp1=-1, yp2=0, xp2=1, yp2=0')
        fig.colorbar(im, ax=ax.ravel().tolist())
        wandb.log({"brt": fig}, step=epoch)
        plt.close(fig)

    if args.validate:
        # ckpt_dir = os.path.join(args.logging_root, args.experiment_name)
        # ckpt = checkpoints.restore_checkpoint(ckpt_dir, target=ckpt)
        val_fn(state, 0)
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
    p.add_argument("--t-max", type=float, default=2.0, required=False, 
                    help="End time of the simulation"
    )
    # initial value function
    p.add_argument("--collision-r", type=float, default=0.2, required=False, 
                    help="Collision radius between vehicles",
    )
    # dynamics
    p.add_argument("--num-states", type=int, default=10, required=False,  # 3 *(1e + 2p) + 1
                    help="Number of states in system including time",
    )
    p.add_argument("--angle-alpha", type=float, default=1.2, required=False, 
                    help="Angle alpha coefficient.",
    )
    p.add_argument("--velocity-e", type=float, default=0.22, required=False, 
                    help="Velocity of Evader",
    )
    p.add_argument("--velocity-p", type=float, default=0.14, required=False, 
                    help="Velocity of Persuer",
    )
    p.add_argument("--omega-e", type=float, default=2.84, required=False, 
                    help="Turn Rate of Evader") 
    p.add_argument("--omega-p", type=float, default=2.00, required=False, 
                    help="Turn Rate of Persuer") 
    # fmt: on
    args = p.parse_args()

    assert args.pretrain_end < args.epochs

    return args

if __name__ in "__main__":
    args = get_args()

    if args.wandb:
        wandb.init(
            project="deepreach_jax", entity="mlu", config=vars(args), save_code=True, tags=['1E2P']
        )
    main(args)
