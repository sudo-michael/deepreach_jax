import jax
import jax.numpy as jnp


def xy_grid(grid_points, x_max=1, y_max=1):
    """Createes a grid of size (grid_points**2, 2) in the range of [-x_max, x_max] x [-y_max, y_max]"""
    x_grid_space = jnp.linspace(-x_max, x_max, num=grid_points)
    y_grid_space = jnp.linspace(-x_max, y_max, num=grid_points)
    X, Y = jnp.meshgrid(x_grid_space, y_grid_space, indexing="ij")
    xy_grid = jnp.column_stack((X.flatten(), Y.flatten()))
    return xy_grid

def create_dataset_sampler(initial_value_function, num_states):
    """A function that returns a dataset sampler for the given initial value function and number of states.

    Args:
        initial_value_function: vmap'd function that takes in tcoords and returns l(x) 
        num_states: number of states in system, excluding time 
        batch_size: number of samples to generate per call (necessary for jit)
    """

    def dataset_sampler(key, dataset_state):
        coords_key, time_key = jax.random.split(key)
        # subtract 1 since we're going to generate a seperate time column
        coords = jax.random.uniform(
            coords_key, (dataset_state.batch_size, num_states - 1), minval=-1, maxval=1
        )

        if dataset_state.counter < dataset_state.pretrain_end:
            time = jnp.ones((dataset_state.batch_size, 1)) * dataset_state.t_min
        else:
            time = (
                dataset_state.t_min
                + jax.random.uniform(
                    time_key,
                    (dataset_state.batch_size, 1),
                    minval=0,
                    maxval=(dataset_state.t_max - dataset_state.t_min)
                    * jnp.minimum(
                        (dataset_state.counter / dataset_state.time_curriculum_end), 1.0
                    ),  # lerp time up to t_max
                )
                # / t_max  # ensure all inputs are in [-1, 1]
            )

        time = time.at[-dataset_state.samples_at_t_min :].set(dataset_state.t_min)
        normalized_tcoords = jnp.concatenate((time, coords), axis=-1)  # (b, [t c])
        boundary_values = initial_value_function(normalized_tcoords)

        dirichlet_mask = (time == dataset_state.t_min).flatten()

        dataset_state = dataset_state.replace(counter=dataset_state.counter + 1)

        return (
            dataset_state,
            normalized_tcoords,
            {
                "source_boundary_values": boundary_values,
                "dirichlet_mask": dirichlet_mask,
            },
        )

    return dataset_sampler
