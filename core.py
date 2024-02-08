from model import Model
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
        acs = acs.detach()
        if args.gpu:
            acs = acs.cpu()
        acs = acs.numpy()
        m.send(acs)
    m.close()
    states = m.q[:args.samplesperbatch]
    del m.q[:args.samplesperbatch]
    return states


def get_random(s, l):
    ds = s.copy()
    np.random.shuffle(ds)
    ds = ds[:l]

    ds, state = zip(*ds)
    return ds, state


def train_policy(s, ds, ip):
    neglogpx, neglogpy = ip(*zip(*s), p2=True, loss=True)
    targetx, targety = (zip(*(d for d in ds if d is not None)))
    loss = ip.backward(neglogpx, targetx, neglogpy, targety)
    return loss
