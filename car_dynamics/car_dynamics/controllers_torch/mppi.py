import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
from termcolor import colored
from car_dynamics.controllers_torch import BaseController
import numpy as np

class MPPIController(BaseController):
    def __init__(self,
                gamma_mean,
                gamma_sigma,
                discount,
                sample_sigma,
                lam,
                a_mean,
                a_cov,
                n_rollouts,
                H,
                device,
                rollout_fn,
                a_min,
                a_max,
                a_mag, # magnitude
                a_shift, # shift,
                delay, # delay
                len_history, 
                rollout_start_fn,
                debug,
                fix_history,
                num_obs,
                num_actions,
    ):
        
        self.gamma_mean: float = gamma_mean
        self.gamma_sigma: float = gamma_sigma
        self.discount: float = discount
        self.sample_sigma: float = sample_sigma
        self.lam: float = lam
        
        self.a_mean: torch.Tensor = a_mean # a should be normalized to [-1, 1] in dynamics
        self.a_cov: torch.Tensor = a_cov
        self.a_mean_init = a_mean[-1:]
        self.a_cov_init = a_cov[-1:]
        # self.a_init: torch.Tensor = ...
        
        self.n_rollouts: int = n_rollouts
        self.H: int = H # horizon
        self.device = device
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # self.reward_fn = reward_fn
        self.rollout_fn = rollout_fn
        
        self.a_min = a_min
        self.a_max = a_max
        self.a_mag = a_mag
        self.a_shift = a_shift
        self.delay = delay
        # assert self.delay > 0
        self.prev_a = [torch.zeros(self.num_actions).to(device)] * delay
        self.len_history = len_history
        self.step_count = 0
        if self.len_history > 0:
            self.state_hist = torch.zeros((self.len_history, self.num_obs + self.num_actions)).to(device)
            
        self.rollout_start_fn = rollout_start_fn
        self.debug = debug
        self.fix_history = fix_history
        self.x_all = []
        self.y_all = []
            
    def _sample_actions(self, ):
        action_dist = MultivariateNormal(loc=self.a_mean, covariance_matrix=self.a_cov)
        # import pdb; pdb.set_trace()
        # print(self.a_mean.device, self.a_cov.device)
        a_sampled = action_dist.sample((self.n_rollouts,))
        # import pdb; pdb.set_trace()
        for d in range(len(self.a_min)):
            a_sampled[:, :, d] = torch.clip(a_sampled[:, :, d], self.a_min[d], self.a_max[d]) * self.a_mag[d] + self.a_shift[d]
        # import pdb; pdb.set_trace()
        return a_sampled
        
    def _get_rollout(self, state_init, actions, fix_history=False):
        n_rollouts = actions.shape[0]
        state = state_init.unsqueeze(0).repeat(n_rollouts, 1)
        state_list = [state]
        obs_history = self.state_hist.unsqueeze(0).repeat(n_rollouts, 1, 1)
        # reward_rollout = torch.zeros((self.n_rollouts), device=self.device)
        self.rollout_start_fn()
        for step in range(actions.shape[1]):
            a_rollout = actions[:, step]
            if (not fix_history) or (step == 0):
                obs_history[:, -1, :self.num_obs] = state[:, :self.num_obs].clone()
                obs_history[:, -1, -self.num_actions:] = a_rollout.clone()
            # print(f"action shape, {a_rollout.shape}")
            state, debug_info = self.rollout_fn(obs_history, state, a_rollout, self.debug)
            if self.debug:
                self.x_all += debug_info['x'].tolist()
                self.y_all += debug_info['y'].tolist()
            if not fix_history:
                obs_history[:, :-1] = obs_history[:, 1:].clone()
            state_list.append(state)
            # import pdb; pdb.set_trace()
            # reward_rollout += reward_fn(state, a_rollout) * (self.discount ** step)
        
        return state_list
    
    def feed_hist(self, obs, action):
        state = torch.tensor(obs[:self.num_obs], device=self.device)
        action_tensor = torch.tensor(action[:self.num_actions], device=self.device)
        self.state_hist[-1, :self.num_obs] = state.clone()
        self.state_hist[-1, self.num_obs:self.num_obs + self.num_actions] = action_tensor.clone()
        self.state_hist[:-1] = self.state_hist[1:].clone()
        
    def __call__(
        self,
        obs,
        reward_fn,
        vis_optim_traj=False,
        use_nn = False,
        vis_all_traj = False,
    ):
        st = time.time()
        a_sampled_raw = self._sample_actions() # Tensor
        # print("sample time", time.time() - st)
        
        ## Delay
        st = time.time()
        
        a_sampled = torch.zeros((self.n_rollouts, self.H + self.delay, a_sampled_raw.shape[2])).to(self.device)
        for i in range(self.delay):
            a_sampled[:, i, :] = self.prev_a[i - self.delay]
        a_sampled[:, self.delay:, :] = a_sampled_raw
        ########
        
        state_init = torch.Tensor(obs).to(self.device)
            
        # import pdb; pdb.set_trace()
        state_list = self._get_rollout(state_init, a_sampled, self.fix_history) # List
        reward_rollout = reward_fn(state_list, a_sampled, self.discount) # Tensor
        cost_rollout = -reward_rollout

        # print("rollout time", time.time() - st)

        cost_exp = torch.exp(-(cost_rollout - torch.min(cost_rollout)) / self.lam)
        weight = cost_exp / cost_exp.sum()
        # import pdb; pdb.set_trace()
        
        a_sampled = a_sampled[:, self.delay:, :]
        self.a_mean = torch.sum(weight[:, None, None] * a_sampled, dim=0) * self.gamma_mean + self.a_mean * (1 - self.gamma_mean)
        # import pdb; pdb.set_trace()
        
        self.a_cov = torch.sum(
                        weight[:, None, None, None] * ((a_sampled - self.a_mean)[..., None] * (a_sampled - self.a_mean)[:, :, None, :]),
                        dim=0,
                    ) * self.gamma_sigma + self.a_cov * (1 - self.gamma_sigma)
        
        u = self.a_mean[0]
        
        optim_traj = None
        if vis_optim_traj:
            if use_nn:
                
                print(colored(f"state init: {state_init}", "green"))
                optim_traj = torch.vstack(self._get_rollout(state_init, self.a_mean.unsqueeze(0), self.fix_history)).detach().cpu().numpy()
                
                # import pdb; pdb.set_trace()
                # print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                if np.abs(optim_traj[-1, 0]) > 10:
                    import pdb; pdb.set_trace()
                    
            else:
                optim_traj = torch.stack(self._get_rollout(state_init, self.a_mean.unsqueeze(0).repeat(self.n_rollouts, 1, 1), self.fix_history))[:, 0, :].detach().cpu().numpy()
                # import pdb; pdb.set_trace()
                print(colored(f"optimal tra (-1): {optim_traj[-1, :2]}" , "red"))
                
                if np.abs(optim_traj[-1, 0]) > 10:
                    import pdb; pdb.set_trace()
                    
                
        # import pdb; pdb.set_trace()
        
        self.a_mean = torch.cat([self.a_mean[1:], self.a_mean[-1:]], dim=0)
        self.a_cov = torch.cat([self.a_cov[1:], self.a_cov[-1:]], dim=0)
        # self.a_mean = torch.cat([self.a_mean[1:], self.a_mean_init], dim=0)
        # self.a_cov = torch.cat([self.a_cov[1:], self.a_cov_init], dim=0)
        # print(self.a_mean_init)
        self.prev_a.append(u)
        # print("mppi time", time.time() - st)
        self.step_count += 1
        info_dict = {'trajectory': optim_traj, 'action': self.a_mean.detach().cpu().numpy(), 'action_candidate': a_sampled.detach().cpu().numpy(), 'x_all': self.x_all, 'y_all': self.y_all} 
        if vis_all_traj:
            info_dict['all_trajectory'] = torch.stack(state_list).detach().cpu().numpy()
            # info_dict['all_trajectory'] = state_list
        return u, info_dict
