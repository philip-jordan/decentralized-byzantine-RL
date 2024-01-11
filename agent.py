import numpy as np
from worker import Worker
from models import *
from environments import *
import json
import hashlib
from pathlib import Path
from options import EXPERIMENT_DIR, EPS_DIAM
from utils import min_diam_subset


class Agent:
    def __init__(self, opts):
        self.opts = opts
        self.workers = [
            Worker(i, True if i < opts.num_byz else False, opts)
            for i in range(opts.num_workers)
        ]
        print(f'{opts.num_workers} workers initialized with {opts.num_byz if opts.num_byz > 0 else "none"} of them Byzantine.')

    def train(self):
        n_traj, avg_rwd = 0, []

        while True:
            if n_traj == 0 or np.random.choice([True, False], p=[self.opts.p, 1-self.opts.p]):
                avgs = [w.sample_grads_large() for w in self.workers]
                honest_avgs = [w for i, w in enumerate(avgs) if not self.workers[i].is_byz]
                avg_rwd += [np.mean(honest_avgs)] * self.opts.N
                n_traj += self.opts.N
                print("Trajectories [{}], avg. reward {}, all: {}".format(n_traj, np.mean(honest_avgs), honest_avgs), flush=True)
            else:
                avgs = [w.sample_grads_small() for w in self.workers]
                honest_avgs = [w for i, w in enumerate(avgs) if not self.workers[i].is_byz]
                avg_rwd += [np.mean(honest_avgs)] * self.opts.B
                n_traj += self.opts.B

            # gradient aggregation
            grads = [w.get_broadcast_grads() for w in self.workers]

            if self.opts.attack_type == 'avg_zero':
                honest_grads = [gs[0] for i, gs in enumerate(grads) if not self.workers[i].is_byz]
                byz_grad = -sum(honest_grads) / self.opts.num_byz
                for i, w in enumerate(self.workers):
                    if w.is_byz:
                        grads[i] = [byz_grad for _ in range(self.opts.num_workers)]

            for w, gs in zip(self.workers, list(zip(*grads))):
                w.set_grads_by_aggregation(gs)

            for w in self.workers:
                w.step()
            
            if self.opts.agreement_type != 'none':
                # parameter agreement
                while True:
                    params = [w.get_broadcast_params() for w in self.workers]
                    for w, ps in zip(self.workers, list(zip(*params))):
                        w.set_params_by_agreement(ps)
                    diam, _ = min_diam_subset(params, len(params)-self.opts.num_byz) if len(params) > 1 else (0, None)
                    if diam < EPS_DIAM:
                        break

            if n_traj >= self.opts.max_trajectories:
                break
        
        return avg_rwd
    

def write_results(rew_hist, opts):
    opts_dict = vars(opts).copy()
    del opts_dict['seed']
    experiment_id = hashlib.sha256(json.dumps(opts_dict, sort_keys=True).encode('utf-8')).hexdigest()[:14]

    print('Saving experiment with ID', experiment_id, '...')
    exp_path = Path(EXPERIMENT_DIR) / experiment_id / ('seed_' + str(opts.seed))
    path = str(exp_path)
    if exp_path.exists():
        print("Experiemnt already conducted.")
        quit()
    exp_path.mkdir(parents=True, exist_ok=True)
    with open(Path(EXPERIMENT_DIR) / experiment_id / 'config.json', 'w') as f:
        opts_dict['experiment_id'] = experiment_id
        json.dump(opts_dict, f, indent=4)
    
    np.save(path + '/rew_hist.npy', np.array(rew_hist))