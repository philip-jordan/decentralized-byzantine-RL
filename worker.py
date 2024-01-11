from attacks import attack_transform
import torch
from models import *
import torch.utils.data
import numpy as np
from agreement import *
from utils import *
from aggregation import *
from attacks import *
from functools import reduce
import operator
from environments import *
import torch.optim as optim

TINY = 10**-9


class Worker:
    def __init__(self, id, is_byz, opts):
        self.id = id
        self.is_byz = is_byz
        self.opts = opts
        self.theta_snap, self.v_old = None, None

        if self.opts.env == 'CartPole':
            self.env = CartPole(is_byz, opts, render=False, seed=self.opts.seed+self.id)
        elif self.opts.env == 'LunarLander':
            self.env = LunarLander(is_byz, opts, render=False, seed=self.opts.seed+self.id)
        else:
            raise Exception("The selected environment is not available.")
        self.env.env._max_episode_steps = self.opts.max_episode_steps
        self.env.set_seed()
        self.policy = Neural_SoftMax(neuralnet(
            self.env.state_space[1],
            self.env.action_space[1],
            self.opts.hidden_layers,
            activation=nn.Tanh()),
            self.env.action_space[2]
        )
        self.param_names, self.param_shapes = zip(*[(n, p.shape) for n, p in self.policy.neural_net.named_parameters()])
#       stdv = 1. / math.sqrt(param.size(-1)) TODO: initialization # param.data.uniform_(-stdv, stdv)
        self.optimizer = optim.Adam(self.policy.neural_net.parameters(), lr=self.opts.lr)
    
    def get_flattened_params(self):
        return np.hstack([torch.flatten(p.detach().clone()).numpy() for p in self.policy.neural_net.parameters()])

    def get_flattened_grads(self):
        return np.hstack([torch.flatten(p.grad.clone()).numpy() for p in self.policy.neural_net.parameters()])

    def load_params_from_flattened(self, flat_params):
        if flat_params is None:
            return
        splits = [reduce(operator.mul, s, 1) for s in self.param_shapes]
        split_params = torch.split(torch.from_numpy(flat_params), splits)
        params = {n: torch.reshape(p, s) for n, p, s in zip(self.param_names, split_params, self.param_shapes)}
        for name, param in self.policy.neural_net.named_parameters():
            param.data = params[name]

    def load_grads_from_flattened(self, flat_grads):
        if flat_grads is None:
            return
        splits = [reduce(operator.mul, s, 1) for s in self.param_shapes]
        split_grads = torch.split(torch.from_numpy(flat_grads), splits)
        grads = {n: torch.reshape(p, s) for n, p, s in zip(self.param_names, split_grads, self.param_shapes)}
        for name, param in self.policy.neural_net.named_parameters():
            param.grad = grads[name]

    def sample_grads_small(self):
        trajectories = self.env.simulate(self.opts.B, policy=self.policy)
        self.vt(trajectories['states'], trajectories['actions'], trajectories['rewards'], self.policy, self.theta_snap)
        tot_reward = sum([sum(trajectories["rewards"][i]) for i in range(self.opts.B)])
        avg_rwd = tot_reward / self.opts.B
        return avg_rwd

    def sample_grads_large(self):
        trajectories = self.env.simulate(self.opts.N, policy=self.policy)
        loss, tot_reward = self.loss(trajectories['states'], trajectories['actions'], trajectories['rewards'], self.policy)
        loss.backward() 
        self.v_old = self.get_flattened_grads()
        avg_rwd = tot_reward / self.opts.N
        return avg_rwd
    
    def step(self):
        self.optimizer.step()
#        self.optimizer.zero_grad()
        for p in self.policy.neural_net.parameters():
            p.grad.data.zero_()
        self.theta_snap = self.get_flattened_params()

    def reset_grad(self):
        for par in self.policy.neural_net.parameters():
            par.grad.data.zero_()

    def loss(self, states, actions, rewards, policy, weights=None):
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*self.R_togo(r_n)).sum(dim=0)
            tot_rwd += r_n.sum().item()
        if weights is not None:
            logprob = logprob*weights

        return -logprob.mean(), tot_rwd
    
    def vt(self, states, actions, rewards, policy, theta_snap):
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        grad_new = self.get_flattened_grads()
        theta_new = self.get_flattened_params()
        self.load_params_from_flattened(theta_snap)
        self.reset_grad()
        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)
        self.load_params_from_flattened(theta_snap)

        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        grad_old = self.get_flattened_grads()
        self.load_params_from_flattened(theta_new)
        self.reset_grad()

        self.load_grads_from_flattened(self.v_old + grad_new - grad_old)
        self.v_old = self.v_old + grad_new - grad_old

    def R_tau(self, rewards):
        return torch.tensor([self.opts.gamma**i*rewards[i] for i in range(rewards.shape[0])]).sum().item()
        
    def R_togo(self, rewards):
        Rewards = rewards.repeat((rewards.shape[0], 1))
        gamma = torch.tensor([self.opts.gamma**i for i in range(rewards.shape[0])])
        Gamma = torch.triu(gamma.repeat((gamma.shape[0], 1)))
        return (Gamma*Rewards).sum(dim=1)

    def importance_weights(self, states, actions, theta_new, theta_snap, policy):
        p_new = [1]*len(states)
        p_old = [1]*len(states)
        self.load_params_from_flattened(theta_new)

        for i, s in enumerate(states):
            acts = actions[i]
            policy.distribution(torch.tensor(s))
            distr_ = policy.policy
            for j, a in enumerate(acts):
                p_new[i] = p_new[i]*distr_[j, a]

        self.load_params_from_flattened(theta_snap)

        for i, s in enumerate(states):
            acts = actions[i]
            policy.distribution(torch.tensor(s))
            distr_ = policy.policy
            for j, a in enumerate(acts):
                p_old[i] = p_old[i]*distr_[j, a]
        return torch.tensor([(p_old[j]+0*TINY)/(p_new[j]+TINY) for j, _ in enumerate(p_new)]).detach()

    def get_broadcast_grads(self):
        grads = self.get_flattened_grads()
        if self.is_byz:
            return attack_transform(grads, self.opts.attack_type, self.opts.num_workers)
        return np.array([grads.copy() for _ in range(self.opts.num_workers)])

    def get_broadcast_params(self):
        params = self.get_flattened_params()
        if self.is_byz:
            return attack_transform(params, self.opts.attack_type, self.opts.num_workers)
        return np.array([params.copy() for _ in range(self.opts.num_workers)])

    def set_params_by_agreement(self, input_params, byz_dir=None):
        if self.opts.agreement_type == 'none':
            return
        elif self.opts.agreement_type == 'avg':
            agreement = simple_average_agree(input_params, self.opts.num_byz)
        elif self.opts.agreement_type == 'mda':
            agreement = mda_average_agree(input_params, self.opts.num_byz, byz_dir, self.opts)
        elif self.opts.agreement_type == 'rbtm':
            agreement = rbtm_average_agree(input_params, self.opts.num_byz)
        elif self.opts.agreement_type == 'h-mda':
            agreement = heuristic_mda_average_agree(input_params, self.opts.num_byz)
        elif self.opts.agreement_type == 'centralized':
            agreement = input_params[-1]
        else:
            raise NotImplementedError
        self.load_params_from_flattened(agreement)

    def set_grads_by_aggregation(self, input_grads):
        if len(input_grads) == 1:
            aggregate = input_grads[0]
        elif self.opts.aggregation_type == 'none':
            return
        elif self.opts.aggregation_type == 'rfa':
            aggregate = rfa_aggregate(input_grads)
        elif self.opts.aggregation_type == 'cm':
            aggregate = cm_aggregate(input_grads)
        elif self.opts.aggregation_type == 'avg':
            aggregate = avg_aggregate(input_grads)
        elif self.opts.aggregation_type == 'krum':
            aggregate = krum_aggregate(input_grads, self.opts)
        else:
            raise NotImplementedError
        self.load_grads_from_flattened(aggregate)