from models.modelv4 import ModelWrapper
import args
import numpy as np
import torch
import tetris


def initial_policy():
    x=tetris.Container()
    a=x.get_shapes()
    b=x.get_state()
    return ModelWrapper(a,torch.tensor([b], dtype=torch.int,device=torch.device("cuda")))

def random_play(m):
    while len(m.q) < 1000:
        m.recv()
        m.send(np.random.randint(10,size=(args.nenvs,2)))
    m.close()
    states = m.q[:]
    del m.q[:]
    return states

def sample_self_play(ip, m,samplesperbatch):
    while len(m.q) < samplesperbatch:
        m.recv()
        with torch.no_grad():

            acs = ip(torch.tensor(np.array(m.data).reshape(-1,500), dtype=torch.int,device=torch.device("cuda")), p2=True)
        acs=acs.cpu().numpy()
        m.send(acs.argmax(-1))
    m.close()
    states = m.q[:]
    del m.q[:]
    return states


def get_random(ds, l):
    state=ds[0].copy()
    l=min(l,len(state))
    acs=ds[1].copy()
    np.random.shuffle(state)
    np.random.shuffle(acs)

    return state[:l],acs[:l]

def train_policy(s, acs, ip):
    neglogpx, neglogpy = ip(s, p2=True, loss=True)
    targetx, targety = (zip(*(d for d in acs if d is not None)))
    loss = ip.backward(neglogpx, targetx, neglogpy, targety)
    return loss
