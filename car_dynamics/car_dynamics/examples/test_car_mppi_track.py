"""
MPPI for goal reaching ...
"""

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from car_dynamics.envs import make_env, KinematicBicycleModel, KinematicParams
from car_dynamics.controllers_torch import MPPIController

DT = .125
VEL = 1.0
N_ROLLOUTS = 10000
H = 4
SIGMA = 1.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = make_env('car-base-single')
trajectory_type = "oval"

SPEED = 1.0

def reward_fn(goal_list: torch.Tensor):
    def reward(state, action, discount):
        """
            - state: contains current state of the car
        """
        # import pdb; pdb.set_trace()
        num_rollouts = action.shape[0]
        horizon = action.shape[1]
        reward_rollout = torch.zeros((num_rollouts), device=action.device)
        reward_activate = torch.ones((num_rollouts), device=action.device)
        
        # import pdb; pdb.set_trace()
        for h in range(horizon):
            
            state_step = state[h+1]
            action_step = action[:, h]
            # import pdb; pdb.set_trace()
            dist = torch.norm(state_step[:, :2] - goal_list[h, :2], dim=1)
            # vel_direction = state[h][:,:2] - state[h-1][:,:2]
            # pos_direction = - state[h][:,:2] + goal_list[h, :2] 
            # dot_product = (vel_direction * pos_direction).sum(dim=1)
            # cos_angle = dot_product / (torch.norm(pos_direction, dim=1) * torch.norm(vel_direction, dim=1) + 1e-7)
            vel_diff = torch.norm(state_step[:, 3:4] - SPEED, dim=1)
            reward = -dist - 0.05 * vel_diff - 0.0 * torch.norm(action_step[:, 1:2], dim=1)
            # reward = - 0.4 * dist - 0.0 * torch.norm(action_step, dim=1) - 0.0 * vel_diff - 0.1 * torch.log(1 + dist)
            # reward = - 0.4 * dist
            reward_rollout += reward *(discount ** h) * reward_activate
        return reward_rollout
    return reward

model_params = KinematicParams(
                    num_envs=N_ROLLOUTS,
                    last_diff_vel=torch.zeros([N_ROLLOUTS, 1]).to(DEVICE),
                    KP_VEL=7.,
                    KD_VEL=.02,
                    MAX_VEL=5.,
                    PROJ_STEER=.34,
                    SHIFT_STEER=0.,
)   


dynamics = KinematicBicycleModel(model_params, device='cpu')


model_params_single = KinematicParams(
                    num_envs=1,
                    last_diff_vel=torch.zeros([1, 1]).to(DEVICE),
                    KP_VEL=7.,
                    KD_VEL=.02,
                    MAX_VEL=5.,
                    PROJ_STEER=.34,
                    SHIFT_STEER=0.,
)   


dynamics_single = KinematicBicycleModel(model_params_single, device='cpu')


dynamics.reset()
dynamics_single.reset()


def rollout_fn(state, action):
    next_state = dynamics.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], 
                               action[:, 0], action[:, 1])
    # import pdb; pdb.set_trace()
    return torch.stack(next_state, dim=1)

sigmas = torch.tensor([SIGMA] * 2)
a_cov_per_step = torch.diag(sigmas**2)
a_cov_init = a_cov_per_step.unsqueeze(0).repeat(H, 1, 1)
# a_cov_prev =  torch.full((H, 2, 2), 3.0**2) * torch.eye(2).unsqueeze(0).repeat(H, 1, 1)

# import pdb; pdb.set_trace()

mppi = MPPIController(
    gamma_mean=1.0,
    gamma_sigma=0.0,
    discount=0.99,
    sample_sigma = 0.5,
    lam = 0.01,
    a_mean=torch.zeros(H, 2, device=DEVICE),
    a_cov = a_cov_init,
    n_rollouts=N_ROLLOUTS,
    H=H,
    device=DEVICE,
    rollout_fn=rollout_fn,
    a_min = [-1., -1],
    a_max = [1., 1.],
    a_mag = [1., 1.],
    a_shift= [0., 0.],
    delay=0
)
done = False
frames = []


obs = env.reset()

def reference_traj(t):
    if trajectory_type == 'circle':
        
        # global total_angle
        center_circle = (.8, 1.2)
        circle_radius = 1.2
        angle = -np.pi/2  - circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
        
    elif trajectory_type == 'counter circle':
        
        # global total_angle
        center_circle = (.9, 1.2)
        circle_radius = 1.2
        angle = -np.pi/2  + circle_radius * SPEED * t
        return np.array([center_circle[0] + circle_radius * np.cos(angle),
                            center_circle[1] + circle_radius * np.sin(angle)])
    elif trajectory_type == 'oval':
        
        center = (0.9, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  - x_radius * SPEED * t

        x = center[0] + x_radius * np.cos(angle)
        y = center[1] + y_radius * np.sin(angle)

        return np.array([x, y])

    elif trajectory_type == 'counter oval':
        center = (0.8, 1.0)
        x_radius = 1.2
        y_radius = 1.4

        # Assuming t varies from 0 to 2π to complete one loop around the oval
        angle = -np.pi/2  + x_radius * SPEED * t

        x = center[0] + x_radius * np.cos(angle)
        y = center[1] + y_radius * np.sin(angle)

        return np.array([x, y])
    else:
        raise NotImplementedError






t = 0.0
goal_list = []
target_list = []
action_list = []
mppi_action_list = []
obs_list = []

waypoint_t_list = np.arange(-np.pi*2-DT, np.pi*2+DT, 0.01)
waypoint_list = np.array([reference_traj(t) for t in waypoint_t_list])
        
while not done:    
    distance_list = np.linalg.norm(waypoint_list - env.obs_state()[:2], axis=-1)
    t_idx = np.argmin(distance_list)
    t_closed = waypoint_t_list[t_idx]
    target_pos_list = [reference_traj(t_closed + i*DT) for i in range(H+1)]
    target_list.append(target_pos_list)
    target_pos_tensor = torch.Tensor(target_pos_list).to(DEVICE).squeeze(dim=-1)
    dynamics.reset()
    # action, mppi_info = mppi(obs, reward_fn(target_pos_tensor))
    action, mppi_info = mppi(env.obs_state(), reward_fn(target_pos_tensor))

    pred_obs = dynamics_single.single_step_numpy(env.obs_state(), action.numpy())
    print("pred", "input", env.obs_state(), action.numpy(), "output", pred_obs)


    obs_list.append(env.obs_state())
    obs, reward, done, info = env.step(action)
    
    print("new obs", env.obs_state())
    # print("real", env.pos[0], env.pos[1], env.yaw, env.vel)
    # frames.append(env.render(mode='rgb_array'))
    # dynamics.params.MAX_VEL += np.random.uniform(-.5, .5)
    # dynamics.params.PROJ_STEER += np.random.uniform(-.2, .2)
    # dynamics.params.SHIFT_STEER += np.random.uniform(-.1, .1)
    # import pdb; pdb.set_trace()
    action_list.append(action.numpy())
    mppi_action_list.append(mppi_info['action'].cpu().numpy().tolist())
    t += .05
    if t > 10.0:
        break
    
    
obs_list = np.array(obs_list)
target_list = np.array(target_list)
action_list = np.array(action_list)
mppi_action_list = np.array(mppi_action_list)


with open(f'tmp/{trajectory_type}_log.json', 'w') as f:
    json.dump({'obs':obs_list.tolist(), 'target':target_list.tolist(), 'action':action_list.tolist(),
               'mppi_actions':mppi_action_list.tolist()}, f)
# for i, trajectory in enumerate(trajectory_list):
#     print(i)
#     plt.clf()
#     for state_rollout in trajectory:
#         plt.plot(state_rollout[:, 0], state_rollout[:, 1], alpha=0.1)
#     plt.plot(obs_list[:, 0], obs_list[:, 1])
#     plt.scatter(obs_list[0, 0], obs_list[0, 1], marker='o', label='start')
#     plt.scatter(env.goal[0], env.goal[1], marker='x', label='goal')
#     plt.legend()
#     plt.savefig(f'tmp/traj/{i}.png')
    
