import jax.numpy as jnp
import jax
from modules import SirenNet
from flax.training import train_state
import optax
from inspect import isfunction
from jax.tree_util import register_pytree_node


def jacobian(apply_fn, params, x):
    f = lambda x: apply_fn(params, x)
    V, f_vjp = jax.vjp(f, x)
    (nablaV,) = f_vjp(jnp.ones_like(V))
    return nablaV, V


def initialize_hji_loss(state, min_with, compute_hamiltonian):
    # TODO refactor and return 2 losses
    @jax.jit
    def hji_loss(params, normalized_tcoords, source_boundary_values, dirichlet_mask):
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


def initialize_opt_ctrl_fn(state, compute_opt_ctrl_fn):
    # @jax.jit
    def opt_ctrl_fn(params, normalized_tcoords):
        nablaV, _ = jacobian(state.apply_fn, params, normalized_tcoords)
        return compute_opt_ctrl_fn(nablaV)

    return opt_ctrl_fn


def initialize_train_state(
    key, num_states, num_nl, num_hl, lr, periodic_transform=None
):
    layers = [num_nl for _ in range(num_hl)]
    if isfunction(periodic_transform):
        model = SirenNet(hidden_layers=layers, transform_fn=periodic_transform)
        num_states = periodic_transform(jnp.ones((1, num_states))).shape[-1]
    else:
        model = SirenNet(hidden_layers=layers)
        num_states = num_states

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(
            key,
            jnp.ones((1, num_states)),
        ),
        tx=optax.adam(learning_rate=lr),
    )
    return state


class HJIData:
    def __init__(self, dataset_state) -> None:
        self.dataset_state = dataset_state

    def normalize_tcords(self, unnormalized_tcoord):
        pass

    def unnormalize_tcords(self, normalized_tcoord):
        pass

    def scale_costates(self, dVdx):
        pass

    def _tree_flatten(self):
        # children = (self.x, ) # arrays / dynamic values
        # aux_data = {dict} # static values
        # return (children, aux_data)
        return None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


register_pytree_node(
    HJIData,
    HJIData._tree_flatten,  # tell JAX what are the children nodes
    HJIData._tree_unflatten,  # tell JAX how to pack back into a RegisteredSpecial
)

class Dynamics:
    def __init__(self, velocity_evader, velocity_peruser, omega_evader, omega_persuer) -> None:
        self.velocity_evader = velocity_evader
        self.velocity_persuer = velocity_peruser
        self.omega_evader = omega_evader
        self.omega_persuer = omega_persuer

    @jax.vmap
    def hamiltonian(self, nablaV, normalized_tcoords):
        # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
        # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].
        tcoords = self.unnormalize_tcoords(normalized_tcoords)

        dVdx = nablaV[1:]

        # TODO understand why we scale dVdx
        dVdx = self.scale_costates(dVdx)

        dVdx_e, dVdy_e, dVdtheta_e = dVdx[0], dVdx[1], dVdx[6]
        dVdx_p1, dVdy_p1, dVdtheta_p1 = dVdx[2], dVdx[3], dVdx[7]
        dVdx_p2, dVdy_p2, dVdtheta_p2 = dVdx[4], dVdx[5], dVdx[8]

        x_e, y_e, theta_e = tcoords[1], tcoords[2], tcoords[7]
        x_p1, y_p1, theta_p1 = tcoords[3], tcoords[4], tcoords[8]
        x_p2, y_p2, theta_p2 = tcoords[5], tcoords[6], tcoords[9]

        ham_evader = self.velocity_evader * dVdx_e * jnp.cos(
            theta_e
        ) + self.velocity_evader * dVdy_e * jnp.sin(theta_e)
        ham_evader = ham_evader + self.omega_evader * jnp.abs(dVdtheta_e)

        ham_persuer1 = self.velocity_persuer * dVdx_p1 * jnp.cos(
            theta_p1
        ) + self.velocity_persuer * dVdy_p1 * jnp.sin(theta_p1)
        ham_persuer1 = ham_persuer1 - self.omega_persuer * jnp.abs(dVdtheta_p1)

        ham_persuer2 = self.velocity_persuer * dVdx_p2 * jnp.cos(
            theta_p2
        ) + self.velocity_persuer * dVdy_p2 * jnp.sin(theta_p2)
        ham_persuer2 = ham_persuer2 - self.omega_persuer * jnp.abs(dVdtheta_p2)

        ham = ham_evader + ham_persuer1 + ham_persuer2

        return ham

    def opt_ctrl_dstb(self, nablaV):
        # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
        # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].
        dVdx = nablaV[1:]
        dVdtheta_e = dVdx[6]
        dVdtheta_p1 = dVdx[7]
        dVdtheta_p2 = dVdx[8]

        # opt_ctrl to maximize V
        opt_ctrl = self.omega_evader * (
            dVdtheta_e > 0
        ) + -self.omega_evader * (dVdtheta_e <= 0)

        # opt_dstb to maximize V
        opt_dstb_p1 = -self.omega_persuer * (
            dVdtheta_p1 > 0
        ) + self.omega_persuer * (dVdtheta_p1 <= 0)
        opt_dstb_p2 = -self.omega_persuer * (
            dVdtheta_p2 > 0
        ) + self.omega_persuer * (dVdtheta_p2 <= 0)

        return (opt_ctrl,), (opt_dstb_p1, opt_dstb_p2)


    def normalize_tcords(self, unnormalized_tcoord):
        pass

    def unnormalize_tcords(self, normalized_tcoord):
        pass

    def scale_costates(self, dVdx):
        pass

    def _tree_flatten(self):
        # children = (self.x, ) # arrays / dynamic values
        # aux_data = {dict} # static values
        # return (children, aux_data)
        return None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


# register_pytree_node(
#     HJIData,
#     HJIData._tree_flatten,  # tell JAX what are the children nodes
#     HJIData._tree_unflatten,  # tell JAX how to pack back into a RegisteredSpecial
# )