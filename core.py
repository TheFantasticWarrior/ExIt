from model import Model
import args
import numpy as np
from torch import no_grad
import tetris


def initial_policy():
    x=tetris.Container()
    a=x.get_shapes()
    return Model(a)

def random_play(m):
    while len(m.q) < args.samplesperbatch:
        m.recv()
        m.send(np.random.randint(10,size=(args.nenvs,2)))
    m.close()
    states = m.q[:]
    del m.q[:]
    return states

def sample_self_play(ip, m):
    while len(m.q) < args.samplesperbatch:
        m.recv()
        with no_grad():
            acs = ip(np.array(m.data).reshape(-1,500), p2=True)
        acs = acs.detach()
        if args.gpu:
            acs = acs.cpu()
        acs = acs.numpy()
        m.send(acs)
    m.close()
    states = m.q[:]
    del m.q[:]
    return states


def get_random(ds, l):
    state=ds[0].copy()
    acs=ds[1].copy()
    np.random.shuffle(state)
    np.random.shuffle(acs)

    return state,acs

def train_policy(s, acs, ip):
    neglogpx, neglogpy = ip(s, p2=True, loss=True)
    targetx, targety = (zip(*(d for d in acs if d is not None)))
    loss = ip.backward(neglogpx, targetx, neglogpy, targety)
    return loss
