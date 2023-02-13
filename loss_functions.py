import jax.numpy as jnp
import jax
def jacobian(apply_fn, params, x):
    f = lambda x: apply_fn(params, x)
    V, f_vjp = jax.vjp(f, x)
    (nablaV,) = f_vjp(jnp.ones_like(V))
    return nablaV, V

def initialize_hji_air3D(state, dataset_state, min_with):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset_state.velocity
    omega_max = dataset_state.omega_max
    alpha_angle = dataset_state.angle_alpha

    @jax.jit
    def hji_air3D_loss(params, tcoords, source_boundary_values, dirichlet_mask):
        # tcoords (b, 4)
        # sbv (b, )
        # dm (b, )
        @jax.vmap
        def hamiltonion(nablaV, tcoords):
            dVdt = nablaV[0]
            dVdx = nablaV[1:]
            
            x_theta = tcoords[3] * 1.0
            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dVdx = dVdx.at[2].divide(alpha_angle)
            # Scale the coordinates
            x_theta = alpha_angle * x_theta

            # Air3D dynamics
            # \dot x    = -v_a + v_b \cos \psi + a y
            # \dot y    = v_b \sin \psi - a x
            # \dot \psi = b - a

            # Compute the hamiltonian for the ego vehicle
            ham = omega_max * jnp.abs(dVdx[0] * tcoords[2] - dVdx[1] * tcoords[1] - dVdx[2])  # Control component
            ham = ham - omega_max * jnp.abs(dVdx[2])  # Disturbance component
            ham = ham + (velocity * (jnp.cos(x_theta) - 1.0) * dVdx[0]) + (velocity * jnp.sin(x_theta) * dVdx[1])  # Constant component
            return ham
        
        nablaV, V = jacobian(state.apply_fn, params, tcoords)
        ham = hamiltonion(nablaV, tcoords)

        # If we are computing BRT then take min with zero
        if min_with == 'zero':
            ham = jnp.clamp(ham, max=0.0)

        dVdt = nablaV[:, 0]
        diff_constraint_hom = dVdt.flatten() - ham

        if min_with == 'target':
            diff_constraint_hom = jnp.maximum(
                diff_constraint_hom, V.flatten() - source_boundary_values
            )

        dirichlet = dirichlet_mask * (V.flatten() - source_boundary_values)

        loss1 = (
            jnp.abs(dirichlet).sum() * V.shape[0] / 15e2
        )  
        loss2 = (
            jnp.abs(jnp.invert(jnp.all(dirichlet_mask)) * diff_constraint_hom).sum()
        )  # if all of dirichlet_max is True, then don't use loss2

        return loss1 + loss2

    return hji_air3D_loss
