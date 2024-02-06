from model import Model
from tree import MCTS
import args
import numpy as np
from torch import no_grad


def initial_policy():
    return Model()


def sample_self_play(ip, m):
    while len(m.q) < args.samplesperbatch:
        state, atk, _ = m.recv()
        with no_grad():
            acs = ip(state, atk, p2=True)
        m.send(acs.cpu())
    m.close()
    states = m.q[:args.samplesperbatch]
    del m.q[:args.samplesperbatch]
    return states


def train_policy(s, ds, ip):
    neglogpx, neglogpy = ip(*zip(*s), p2=True, loss=True)
    print(neglogpx.size())
    targetx, targety = (zip(*(d for d in ds if d is not None)))
    print(len(targetx))
    ip.backward(neglogpx, targetx, neglogpy, targety)
    return ip
