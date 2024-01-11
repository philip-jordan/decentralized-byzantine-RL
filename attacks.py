import numpy as np
from utils import *


def attack_transform(in_vec, attack_type, num_workers):
    if attack_type == 'none':
        return np.array([in_vec.copy() for _ in range(num_workers)])
    elif attack_type == 'random':
        return np.array([np.random.uniform(0, 1, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_unif_-1_1':
        return np.array([np.random.uniform(-1, 1, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_unif_-10_10':
        return np.array([np.random.uniform(-10, 10, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_unif_-100_100':
        return np.array([np.random.uniform(-100, 100, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_unif_-1000_1000':
        return np.array([np.random.uniform(-1000, 1000, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_uniform_as_params':
        low = np.min(in_vec)
        high = np.max(in_vec)
        return np.array([np.random.uniform(low, high, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_normal_as_params':
        mean = np.mean(in_vec)
        std = np.std(in_vec)
        return np.array([np.random.normal(mean, std, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_normal_0_1':
        return np.array([np.random.normal(0, 1, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_normal_0_10':
        return np.array([np.random.normal(0, 10, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_normal_0_100':
        return np.array([np.random.normal(0, 100, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'random_normal_0_1000':
        return np.array([np.random.normal(0, 1000, *in_vec.shape) for _ in range(num_workers)]).astype(np.float32)
    
    ### Attacks used in 'Genuinely dist. ML' paper #############
    elif attack_type == 'reversed':
        return np.array([-in_vec.copy() for _ in range(num_workers)])
    elif attack_type == 'partial_drop_0.1_cons':
        r = partial_drop(in_vec, .1)
        return np.array([r.copy() for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'partial_drop_0.1_incons':
        return np.array([partial_drop(in_vec, .1) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'partial_drop_0.5_cons':
        r = partial_drop(in_vec, .5)
        return np.array([r.copy() for _ in range(num_workers)])
    elif attack_type == 'partial_drop_0.5_incons':
        return np.array([partial_drop(in_vec, .5) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'partial_drop_0.9_cons':
        r = partial_drop(in_vec, .9)
        return np.array([r.copy() for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'partial_drop_0.9_incons':
        return np.array([partial_drop(in_vec, .9) for _ in range(num_workers)]).astype(np.float32)
    
    ### Our attack ideas #######################################
    elif attack_type == 'rand_orth_0_1_cons':
        r = random_orthogonal_to(in_vec)
        return np.array([r.copy() for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'rand_orth_0_1_incons':
        return np.array([random_orthogonal_to(in_vec) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'rand_orth_0_10_cons':
        r = random_orthogonal_to(in_vec, 0, 10)
        return np.array([r.copy() for _ in range(num_workers)])
    elif attack_type == 'rand_orth_0_10_incons':
        return np.array([random_orthogonal_to(in_vec, 0, 10) for _ in range(num_workers)]).astype(np.float32)
    elif attack_type == 'rand_orth_0_100_cons':
        r = random_orthogonal_to(in_vec, 0, 100)
        return np.array([r.copy() for _ in range(num_workers)])
    elif attack_type == 'rand_orth_0_100_incons':
        return np.array([random_orthogonal_to(in_vec, 0, 100) for _ in range(num_workers)]).astype(np.float32)

    elif attack_type in ['random_action',
                         'avg_zero',
                         'random_reward',
                         'reward_flipping',
                         'mda_fooling',
                         'mda_fooling_orth']:
        # handled elsewhere
        return np.array([in_vec.copy() for _ in range(num_workers)])

    else:
        raise NotImplementedError