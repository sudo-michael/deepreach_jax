import jax.numpy as jnp
import jax


def jacobian(apply_fn, params, x):
    f = lambda x: apply_fn(params, x)
    V, f_vjp = jax.vjp(f, x)
    (nablaV,) = f_vjp(jnp.ones_like(V))
    return nablaV, V


def initialize_hji_loss(state, min_with, compute_hamiltonian):

    # TODO refactor and return 2 losses
    @jax.jit
    def hji_loss(
        params, normalized_tcoords, source_boundary_values, dirichlet_mask
    ):
        nablaV, V = jacobian(state.apply_fn, params, normalized_tcoords)
        ham = compute_hamiltonian(nablaV, normalized_tcoords)

        # If we are computing BRT then take min with zero
        if min_with == "zero":
            ham = jnp.clamp(ham, max=0.0)

        dVdt = nablaV[:, 0]
        diff_constraint_hom = dVdt.flatten() - ham

        if min_with == "target":
            diff_constraint_hom = jnp.maximum(
                diff_constraint_hom, V.flatten() - source_boundary_values
            )

        dirichlet = dirichlet_mask * (V.flatten() - source_boundary_values)

        # h_1 loss in deepreach paper
        loss1 = jnp.abs(dirichlet).sum() * V.shape[0] / 15e2

        # h_2 loss in deepreach paper
        # since we're integrating from 0 to T, it should be -D_t in the paper
        # it is imporant to not include this loss during pretraing for some reason
        loss2 = jnp.abs(jnp.invert(jnp.all(dirichlet_mask)) * diff_constraint_hom).sum()

        return loss1 + loss2, {
            "dirichlet_loss": loss1,
            "diff_constraint_hom_loss": loss2,
        }

    return hji_loss
