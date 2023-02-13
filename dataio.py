from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class DatasetState:
    counter: int
    pretrain_end: int  # for only training ground truth value function at t=t_min
    counter_end: int  # when time can finally be sampled at t_max
    batch_size: int
    samples_at_t_min: int
    # this should be somewhere else
    velocity: float
    omega_max: float
    angle_alpha: float

def xy_grid(grid_points):
    """Createes a grid of size (grid_ponits**2, 2) in the range of [-1, 1]
    """
    grid_space = jnp.linspace(0, 1, num=grid_points)
    X, Y = jnp.meshgrid(grid_space, grid_space, indexing="ij")
    xy_grid = jnp.column_stack((X.flatten(), Y.flatten()))
    xy_grid -= 0.5
    xy_grid *= 2.0
    return xy_grid

def create_dubins_3d_dataset_sampler(
    t_min=0, t_max=1, collision_r=0.5, num_states = 3
):
    """
    Create a funcion that sample points from the range:
        [t_min, t_max] x [-1, 1]^n
    A pretrain_limit is used to initially generate points only at time == t_min 
    Otherwise a time vector is sampled uniformally from:
        [t_min , min(counter / lerp_time_end, 1.0) (t_max - t_min) ] / t_max
    Time is normlized to be in the range [0, 1]
    Args:
        t_min: _description_. Defaults to 0.
        t_max: _description_. Defaults to 1.
        num_src_samples: _description_. Defaults to 10_000.
        collision_r: _description_. Defaults to 0.5.
    Returns:
        dataset:
        cooords: input to model 
        {"boundary_values" : boundary_values (l(x)), 
         "diriichlet_mask": dirichlet_mask  (indicator function, where t == t_min)
        }
    """
    # TODO figure how to jit this
    def generate_dubins_3d(key, dataset_state):
        key, coords_key, time_key = jax.random.split(key, num_states)
        coords = jax.random.uniform(coords_key, (dataset_state.batch_size, num_states), minval=-1, maxval=1)

        if dataset_state.counter < dataset_state.pretrain_end:
            time = jnp.ones((dataset_state.batch_size, 1)) * t_min
        else:
            time = (
                t_min
                + jax.random.uniform(
                    time_key,
                    (dataset_state.batch_size, 1),
                    minval=0,
                    maxval=(t_max - t_min)
                    * jnp.minimum(
                        (dataset_state.counter / dataset_state.counter_end), 1.0
                    ),
                )
                # / t_max  # ensure all inputs are in [-1, 1]
            )

        time = time.at[-dataset_state.samples_at_t_min:].set(t_min)
        tcoords = jnp.concatenate((time, coords), axis=-1)  # (b, [t c])

        boundary_values = jnp.linalg.norm(tcoords[:, 1:3], axis=-1) - collision_r

        # (TODO: remove) normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5
        boundary_values = (boundary_values - mean)*norm_to/var

        # TODO figure how to jit this
        # dirichlet_mask == coords at time t=t_min
        if dataset_state.counter < dataset_state.pretrain_end:
            dirichlet_mask = jnp.ones(dataset_state.batch_size)
        else:
            dirichlet_mask = (time == t_min).flatten()

        dataset_state = dataset_state.replace(counter=dataset_state.counter + 1)

        return (
            dataset_state,
            tcoords,
            {"source_boundary_values": boundary_values, "dirichlet_mask": dirichlet_mask},
        )
        

    return generate_dubins_3d
