from core import initial_policy, sample_self_play, train_policy
import args
from mp import manager, TreeMan
import torch.multiprocessing as mp
import numpy as np
import torch

if __name__ == "__main__":
    ip = initial_policy()
    try:
        checkpoint = torch.load(args.path)
        ip.load_state_dict(checkpoint['model_state_dict'])
        ip.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        last_epoch = 0
    ip.to(torch.device("cuda"))
    s = []
    acs = []
    mn = mp.Manager()
    results = mn.list([None]*args.samplesperbatch)
    states = mn.list()
    for i in range(last_epoch, last_epoch+args.loops):
        m = manager(args.nenvs, args.seed, args.render, states)

        s += sample_self_play(ip, m)
        ds = s.copy()
        np.random.shuffle(ds)
        ds = ds[:args.samplesperbatch]

        ds, state = zip(*ds)
        tm = TreeMan(ip, args.nenvs, state, results, 1e5, True)
        while tm.remotes:
            try:
                tm.recv()
            except KeyboardInterrupt:
                breakpoint()
        tm.close()
        train_policy(ds, list(results), ip)
        torch.save({
            'epoch': i,
            'model_state_dict': ip.state_dict(),
            'optim_state_dict': ip.optimizer.state_dict(),
        }, args.path)
        del s[:len(s)-args.max_record]
