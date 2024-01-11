import argparse

EXPERIMENT_DIR = 'experiments/'
EPS_DIAM = 1e-5


def get_options(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_byz', type=int, default=0)
    parser.add_argument('--attack_type', type=str, choices=['none',
                                                            'random_unif_-1000_1000',
                                                            'random_action',
                                                            'avg_zero'], default='none')
    parser.add_argument('--agreement_type', type=str, choices=['mda', 'centralized', 'none'],
                        default='none')
    parser.add_argument('--aggregation_type', type=str, choices=['rfa', 'avg', 'none'],
                        default='none')
    parser.add_argument('--max_trajectories', type=int, default=10000)
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--B', type=int, default=4)
    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--env', type=str, choices=['CartPole', 'LunarLander'], default='CartPole')
    
    opts = parser.parse_args(args)

    # additional setup for RL environments
    if opts.env == 'CartPole':
        opts.max_episode_steps = 500
        opts.max_trajectories = 10
        opts.gamma = 0.999
        opts.hidden_layers = [16, 16]
        opts.N = 50
        opts.B = 4
        opts.p = 0.2

    if opts.env == 'LunarLander':
        opts.max_episode_steps = 1000  
        opts.max_trajectories = 10
        opts.gamma  = 0.999
        opts.hidden_layers = [64, 64]
        opts.N = 96
        opts.B = 32
        opts.p = 0.2

    return opts