To all of you who've successfully defended (proposal or dissertation), how long did it take you to get your equilibrium back?

import jax.numpy as jnp
state = jnp.array([1, 1.0, 1.0, 1.0])
obs = jnp.array([0., 0., 0.])

def controller(t, state, eps=0.1):
    if V(state) < eps:
        u = compute_opt_ctrl()
    else:
        if t % 2 == 0:
            return dataset_state.omega_max
        else:
            return -dataset_state.omega_max

def simulate(state, control, dt=0.05):
    theta = state[3]
    A = jnp.array([
        [1.0],
        [jnp.cos(theta)],
        [jnp.sin(theta)],
        [1.0],
    ])

    b = jnp.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ])

    u = jnp.array([
        [0.0],
        [0.0],
        [0.0],
        [control],
    ])

    return state + (A @ state + b  @ u) * 0.05

    

history = [state]
for t in range(100):
    control = controller(t, state)
    next_state = simulate(state, control)
    history.append(state)


if __name__ in "__main__":
    main(args)