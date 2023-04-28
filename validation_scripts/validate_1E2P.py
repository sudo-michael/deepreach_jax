import itertools
import os
import sys

# it's a bit faster to run on CPU than on GPU
# 0.017 vs 0.021 per step
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import gym
import jax
import jax.numpy as jnp
import numpy as np
from atu3.brt.brt_air_3d import car_brt, grid
from atu3.brt.brt_static_obstacle_3d import goal_r
from atu3.utils import normalize_angle
from flax import struct
from flax.training import train_state
import orbax.checkpoint as orbax
from flax.training import checkpoints

import matplotlib.pyplot as plt
import pickle

# TODO: make this into a package with setuptools so we don't have to do this
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataio import create_dataset_sampler, xy_grid
from experiment_scripts.hji_1E2P import (
    DatasetState,
    create_hj_fn,
    create_normalization_fn,
    create_train_state,
)

from hj_functions import initialize_train_state, initialize_opt_ctrl_fn, jacobian
from modules import SirenNet
from utils import normalize_value_function, unnormalize_value_function


class Air3DNpEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, n, use_hj=False, use_deepreach=False) -> None:
        self.car = car_brt
        # NOTE: let's try making the persuer slower since optimal disturbance makes it hard for the evader to escape
        self.car.vp = 0.11
        self.dt = 0.05
        self.use_hj = use_hj
        self.n = n

        self.action_space = gym.spaces.Box(
            low=-self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,)
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([-10, -10, -1, -1] * (self.n + 1) + [-10, -10]),
            high=np.array([10, 10, 1, 1] * (self.n + 1) + [10, 10]),
            dtype=np.float32,
        )

        self.world_width = 10
        self.world_height = 10

        # state
        self.persuer_states = [np.array([-1.5, -1.5, np.pi / 4]),  np.array([-1.5, -1.5, np.pi / 4])]
        self.evader_state = np.array([1.0, 1.0, np.pi / 4])
        self.goal_location = np.array([1.5, 1.5])
        self.goal_r = goal_r

        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        if use_deepreach:
            self.use_deepreach = use_deepreach
            self.opt_ctrl_dstb_fn, self.value_fn, self.dataset_state = load_deepreach(
                "logs/1e2p_atu3_4"
            )

        else:
            # self.brt = np.load(os.path.join(dir_path, f"assets/brts/air3d_brt_0.npy"))
            # self.backup_brt = np.load(
            # os.path.join(dir_path, f"assets/brts/backup_air3d_brt_0.npy")
            # )
            self.grid = grid

    def step(self, action):
        info = {}
        info["used_hj"] = False
        if self.use_hj and self.use_opt_ctrl():
            action = self.opt_ctrl()
            info["used_hj"] = True

        # self.evader_state = (
        #     self.car.dynamics_non_hcl(0, self.evader_state, action) * self.dt
        #     + self.evader_state
        # )
        # self.evader_state[2] = normalize_angle(self.evader_state[2])
        if self.use_deepreach:
            persuer_actions = self.opt_dstb(self.persuer_states)
            for i in range(self.n):
                self.persuer_states[i] = (
                    self.car.dynamics_non_hcl(
                        0, self.persuer_states[i], persuer_actions[i], is_evader=False
                    )
                    * self.dt
                    + self.persuer_states[i]
                )
                self.persuer_states[i][2] = normalize_angle(self.persuer_states[i][2])
        else:
            for i in range(self.n):
                persuer_action = self.opt_dstb(self.persuer_states[i])
                self.persuer_states[i] = (
                    self.car.dynamics_non_hcl(
                        0, self.persuer_states[i], persuer_action, is_evader=False
                    )
                    * self.dt
                    + self.persuer_states[i]
                )
                self.persuer_states[i][2] = normalize_angle(self.persuer_states[i][2])

        dist_to_goal = np.linalg.norm(self.evader_state[:2] - self.goal_location[:2])
        reward = -dist_to_goal
        done = False

        info["cost"] = 0

        if (
            np.linalg.norm(self.evader_state[:2] - self.goal_location)
            < self.goal_r + self.car.r
        ):
            done = True
            info["collision"] = "goal"
        elif np.any(
            [
                np.linalg.norm(self.evader_state[:2] - self.persuer_states[i][:2])
                < self.car.r * 2
                for i in range(self.n)
            ]
        ):
            done = True
            info["collision"] = "persuer"

        return (
            np.copy(
                self.get_obs(self.evader_state, self.persuer_states, self.goal_location)
            ),
            reward,
            done,
            info,
        )

    def reset(self, seed=None):
        # DEBUG: the following cases show when the brt is not working
        # works
        # self.evader_state = np.array([0.0, 0.0, 0.0])
        # self.persuer_states[0] = np.array([1.0, 0.3, -np.pi])
        # doesn't
        # self.evader_state = np.array([0.0, 0.0, np.pi])
        # works
        self.evader_state = np.array([0.0, 0.0, -np.pi])
        self.persuer_states[0] = np.array([-1.0, 0.0, 0.0])
        self.persuer_states[1] = np.array([1.0, 0.0, -np.pi])
        # ddoesn't
        # self.evader_state = np.array([0.0, 0.0, 0.0])
        self.evader_state = np.array([0.0, -1.0, np.pi/2])
        self.persuer_states[0] = np.array([-1.0, 0.0, 0.0])
        self.persuer_states[1] = np.array([1.0, 0.0, -np.pi])
        goal_locations = [
            np.array([2.5, 2.5]),
            np.array([0, 3.0]),
            np.array([-2.5, 2.5]),
            np.array([3.0, 0]),
            np.array([2.5, -2.5]),
        ]

        random_idx = np.random.randint(0, len(goal_locations))
        self.goal_location = goal_locations[random_idx]

        # for i in range(self.n):
        #     self.persuer_states[i] = np.random.uniform(
        #         low=-self.world_boundary, high=self.world_boundary
        #     )

        #     # insert the persuer between the goal and the evader
        #     self.persuer_states[i][:2] = goal_locations[random_idx] // 2

        info = {}
        info["cost"] = 0
        info["collision"] = "none"

        return self.get_obs(self.evader_state, self.persuer_states, self.goal_location)

    def render(self, mode="human"):
        self.ax.clear()

        def add_robot(state, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=self.car.r, color=color))

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color="blue")
        for i in range(self.n):
            add_robot(self.persuer_states[i], color="red")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)


        # X, Y = np.meshgrid(
        #     np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
        #     np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
        #     indexing="ij",
        # )

        # relative_state = self.relative_state(self.persuer_states[0])
        # index = self.grid.get_index(relative_state)
        # angle = self.evader_state[2] % (2 * np.pi)
        # Xr = X * np.cos(angle) - Y * np.sin(angle)
        # Yr = X * np.sin(angle) + Y * np.cos(angle)

        # DEBUG: visualize relative state
        # add_robot(relative_state, color="orange")
        # add_robot(np.zeros(3), color="yellow")

        if self.use_hj and not self.use_deepreach:
            self.ax.contour(
                Xr + self.evader_state[0],
                Yr + self.evader_state[1],
                # X, Y,
                self.brt[:, :, index[2]],
                levels=[0.1],
            )
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])
            # plt.show()
        return img

    def close(self):
        plt.close()
        return

    def use_opt_ctrl(self, threshold=0.2):
        if self.use_deepreach:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            value = self.value_fn(jnp.array(unnormalized_tcoords))
            return value.item() < threshold
        else:
            relative_state = self.relative_state(self.persuer_states[0])
            return self.grid.get_value(self.brt, relative_state) < threshold

    def opt_ctrl(self):
        if self.use_deepreach:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            opt_ctrl, _ = self.opt_ctrl_dstb_fn(jnp.array(unnormalized_tcoords)) # (1, ), _
            return np.array(opt_ctrl[0]) # (1, )
        elif self.n > 1:
            raise NotImplementedError("Only support 1 persuer for now")
        relative_state = self.relative_state(self.persuer_states[0])
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        # should this be relative_state or self.evader_state?
        # opt_ctrl = self.car.opt_ctrl_non_hcl(self.evader_state, spat_deriv)
        opt_ctrl = self.car.opt_ctrl_non_hcl(relative_state, spat_deriv)
        return opt_ctrl

    def opt_dstb(self, persuer_state):
        if self.use_deepreach:
            unnormalized_tcoords = self.deepreach_state(
                self.evader_state, self.persuer_states
            )
            _, opt_dstbs = self.opt_ctrl_dstb_fn(jnp.array(unnormalized_tcoords))
            return np.array(opt_dstbs)
        if self.n > 1:
            raise NotImplementedError("Only support 1 persuer for now")
        relative_state = self.relative_state(persuer_state)
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        if spat_deriv[2] == 0:
            relative_state = persuer_state - self.evader_state
            relative_state[2] = persuer_state[2]
            index = self.grid.get_index(relative_state)
            spat_deriv = spa_deriv(index, self.backup_brt, self.grid)

        opt_dstb = self.car.opt_dstb_non_hcl(spat_deriv)
        return opt_dstb

    def relative_state(self, persuer_state):
        rotated_relative_state = np.zeros(3)
        relative_state = persuer_state - self.evader_state

        angle = -self.evader_state[2]

        # fmt: off
        # brt assume that evader_state is at theta=0
        rotated_relative_state[0] = relative_state[0] * np.cos(angle) - relative_state[1] * np.sin(angle)
        rotated_relative_state[1] = relative_state[0] * np.sin(angle) + relative_state[1] * np.cos(angle)
        # fmt: on

        # after rotating by -evader_state[2], the relative angle will still be the same
        rotated_relative_state[2] = normalize_angle(relative_state[2])
        # print(rotated_relative_state)
        return rotated_relative_state

    def deepreach_state(self, evader_state, persuer_states):
        state = jnp.expand_dims(
            state_to_unnormalized_tcoords(
                evader_state, persuer_states, t=self.dataset_state.t_max
            ),
            0,
        )
        return state

    def get_obs(self, evader_state, persuer_states, goal):
        # return [x y cos(theta) sin(theta) all evader and persuers states]
        t = (
            [self.theta_to_cos_sin(evader_state)],
            list(map(self.theta_to_cos_sin, persuer_states)),
            [goal[:2]],
        )
        return np.concatenate((tuple(itertools.chain.from_iterable(t))))

    def theta_to_cos_sin(self, state):
        return np.array(
            [state[0], state[1], np.cos(state[2]), np.sin(state[2])], dtype=np.float32
        )


# if __name__ in "__main__":
#     from datetime import datetime

#     run_name = f"debug__{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

#     gym.logger.set_level(10)

#     env = Air3DNpEnv(1, use_hj=True)
#     env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
#     obs = env.reset()
#     print(obs.shape)
#     done = False
#     t = 0
#     while not done:
#         action = env.action_space.sample()
#         if t % 2 == 0:
#             action = np.array([env.unwrapped.car.we_max])
#         else:
#             action = np.array([-env.unwrapped.car.we_max])
#         _, _, _, info = env.step(action)
#         t += 1

#         obs, reward, done, info = env.step(action)
#         # print(f"{reward=}")
#         if done:
#             print(info)
#             print("done")
#             break

#         env.render()
#     env.close()
def state_to_unnormalized_tcoords(evader_state, persuer_states, t=0.0):
    # [state sequence: 0, 1,   2,   3     4,    5,    6,    7,       8,        9].
    # [state sequence: t, x_e, y_e, x_p1, y_p1, x_p2, y_p2, theta_e, theta_p1, theta_p2].
    unnormalized_tcoords = jnp.array(
        [
            t,
            evader_state[0],
            evader_state[1],
            persuer_states[0][0],
            persuer_states[0][1],
            persuer_states[1][0],
            persuer_states[1][1],
            evader_state[2],
            persuer_states[0][2],
            persuer_states[1][2],
        ]
    )
    return unnormalized_tcoords


def load_deepreach(ckpt_dir):
    ckpt_dir = os.path.join("logs", "1e2p_atu3_2")

    with open(os.path.join(ckpt_dir, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    dataset_state = DatasetState(
        counter=-1,
        pretrain_end=-1,
        counter_end=-1,
        batch_size=-1,
        samples_at_t_min=-1,
        t_min=-1,
        t_max=-1,
        velocity_evader=-1,
        velocity_persuer=-1,
        omega_evader=-1,
        omega_persuer=-1,
        collision_r=-1,
    )

    state = create_train_state(
        model_key,
        dataset_state,
        args.num_states,
        args.num_nl,
        args.num_hl,
        args.lr,
        args.periodic_transform,
    )

    ckpt = {"model": state, "config": vars(args), "dataset": dataset_state}
    ckpt = checkpoints.restore_checkpoint(
        os.path.join(ckpt_dir, "checkpoints"), target=ckpt
    )

    state = ckpt["model"]
    dataset_state = ckpt["dataset"]

    (
        initial_value_function,
        compute_hamiltonian,
        compute_opt_ctrl_dstb_fn,
    ) = create_hj_fn(dataset_state)
    (
        unnormalize_tcoords_fn,
        normalize_tcoords_fn,
        scale_costates,
    ) = create_normalization_fn(dataset_state)
    # DEEPREACH

    @jax.jit
    def opt_ctrl_dstb_fn(unnormalized_tcoords: jnp.array):
        normalize_tcoords = normalize_tcoords_fn(unnormalized_tcoords)
        nablaV, _ = jacobian(state.apply_fn, state.params, normalize_tcoords)
        return compute_opt_ctrl_dstb_fn(nablaV)

    def value_fn(unnormalized_tcoords: jnp.array):
        normalize_tcoords = normalize_tcoords_fn(unnormalized_tcoords)
        return state.apply_fn(state.params, normalize_tcoords)

    return opt_ctrl_dstb_fn, value_fn, dataset_state


def main():
    from gym.wrappers import TimeLimit
    import time
    env = Air3DNpEnv(n=2, use_hj=True, use_deepreach=True)
    env = TimeLimit(env, max_episode_steps=500)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        # time function call
        start = time.process_time()
        next_obs, reward, done, info = env.step(action)
        print(f"time : {time.process_time() - start=}")
        # env.render()

    # opt_ctrl_dstb_fn, value_fn, dataset_state = load_deepreach("logs/1e2p_atu3_2")

    # # Get the meshgrid in the (x, y) coordinate
    # grid_points = 200
    # mgrid_coords = xy_grid(
    #     200, x_max=dataset_state.alpha["x"], y_max=dataset_state.alpha["y"]
    # )

    # ones = np.ones((mgrid_coords.shape[0], 1))
    # time_coords = ones * 2.0
    # x_p1 = ones * persuer_states[0][0]
    # y_p1 = ones * persuer_states[0][1]
    # x_p2 = ones * persuer_states[1][0]
    # y_p2 = ones * persuer_states[1][1]
    # theta_evader = ones * evader_state[2]
    # theta_p1 = ones * persuer_states[0][2]
    # theta_p2 = ones * persuer_states[1][2]
    # unnormalized_tcoords = np.concatenate(
    #     (
    #         time_coords,
    #         mgrid_coords,
    #         x_p1,
    #         y_p1,
    #         x_p2,
    #         y_p2,
    #         theta_evader,
    #         theta_p1,
    #         theta_p2,
    #     ),
    #     axis=1,
    # )

    # unnormalized_tcoords = jnp.array(unnormalized_tcoords)
    # V = value_fn(unnormalized_tcoords)

    # V = np.array(V)
    # V = V.reshape((grid_points, grid_points))

    # V = unnormalize_value_function(
    #     V, dataset_state.norm_to, dataset_state.mean, dataset_state.var
    # )

    # # Plot the zero level sets
    # V = (V <= 0.001) * 1.0

    # # Plot the actual data
    # # im = plt.imshow(
    # #     V.T,
    # #     cmap="bwr",
    # #     origin="lower",
    # #     extent=(-1.0, 1.0, -1.0, 1.0),
    # # )

    # # state = jnp.expand_dims(
    # #     state_to_unnormalized_tcoords(evader_state, persuer_states, t=args.t_max), 0
    # # )
    # # _, opt_dstb = opt_dstb_fn(normalize_tcoords(unnormalized_tcoords))
    # # opt_dstb_p1 = opt_dstb[0].item()
    # # opt_dstb_p2 = opt_dstb[1].item()


    # TODO  ensure that both persuers optimal controll good
        # TODO try switching pos of p1 and p2
        # TODO try using periodic function

if __name__ in "__main__":
    main()
