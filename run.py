#!/usr/bin/env python3

import os
import warnings
import torch
import numpy as np
from agent import Agent, write_results
from options import get_options
import random


def run(opts):
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    agent = Agent(opts)
    rew_hist = agent.train()
    write_results(rew_hist, opts)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run(get_options())