import sys
import os
from flax.training import train_state

import jax
import jax.numpy as jnp
import optax

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import SirenNet

model = SirenNet(hidden_layers=[256, 256])
key = jax.random.PRNGKey(0)
model_params = model.init(key, jnp.ones((1, 4)))['params']

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=model_params,
    tx=optax.adam(learning_rate=3e-4),
)


target = jnp.zeros((1, 4))
print(state.apply_fn({'params': state.params}, jnp.ones((1, 4))))
def jacobian(apply_fn, params, x):
    f = lambda x: apply_fn({'params': params}, x)
    V, f_vjp = jax.vjp(f, x)
    (nablaV,) = f_vjp(jnp.ones_like(V))
    return nablaV, V

def loss_fn(params):
    nablaV, v = jacobian(state.apply_fn, params, jnp.ones((1000, 4)))
    # t = jnp.zeros((1000, ))
    # return ((v - t)**2).mean() + nablaV.mean()
    return v.mean()
    
value, grad = jax.jit(jax.value_and_grad(loss_fn))(state.params) 
# print(grad)


# print()
    
value, grad = jax.jit(jax.value_and_grad(loss_fn))(state.params) 
# print(grad)


# print()