import torch
import gym
import random
import numpy as np


torch.backends.cudnn.deterministic=True

class Environment:
    def __init__(self, is_byz, opts, render=False, seed=None):
       self.render = render
       self.env_seed = seed
       self.is_byz = is_byz
       self.opts = opts

    def set_seed(self):
        if self.env_seed is not None:
            self.env.seed(self.env_seed)   
            self.env.action_space.seed(self.env_seed)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def simulate(self, N, policy=None, verbose=False):
        states_n, actions_n, rewards_n = [], [], []
        tot_reward = 0

        for episode in range(N):
            if verbose:
                print("episode {} of {}\n".format(episode+1, N))

            states, actions, rewards = [], [], []
            done = False
            observation = self.env.reset()

            while not done:
                if self.render:
                    self.env.render()   
                states.append(observation.tolist()) 
                
                if policy == None or (self.is_byz and self.opts.attack_type == 'random_action'):                    
                    action = self.env.action_space.sample()
                    action = np.asarray(action)
                else:
                    policy.distribution(torch.tensor([observation], dtype=torch.float32))
                    action = policy.sample()[0].numpy()
                observation, reward, done, _ = self.env.step(action)

                tot_reward += reward

                rewards.append(reward)
                actions.append(action.tolist())

            states_n.append(states)
            actions_n.append(actions)
            rewards_n.append(rewards)

        tot_reward = tot_reward/N    
        if self.is_byz and self.opts.attack_type == 'random_reward':
            tot_reward = random.randint(0, 500)
                
        self.env.close()
        return {"states": states_n, "actions": actions_n, "rewards": rewards_n}


class CartPole(Environment):
    def __init__(self, is_byz, opts, render=False, seed=None):
        super().__init__(is_byz, opts, render, seed)
        self.env = gym.make('CartPole-v1')
        self.state_space = ("Continuous", 4)
        self.action_space = ("Discrete", 2, [0,1])


class LunarLander(Environment):
    def __init__(self, is_byz, opts, render=False, seed=None):
        super().__init__(is_byz, opts, render, seed)
        self.env = gym.make('LunarLander-v2')
        self.state_space = ("Continuous", self.env.observation_space.shape[0])
        self.action_space = ("Discrete", self.env.action_space.n, [0, 1, 2, 3])