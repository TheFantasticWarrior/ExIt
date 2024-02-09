from core import initial_policy, sample_self_play, get_random, train_policy
import args
from mp import manager, TreeMan
import torch.multiprocessing as mp
import torch
import gc
from hanging_threads import start_monitoring

if __name__ == "__main__":
    ip = initial_policy()
    if args.load:
        checkpoint = torch.load(args.path)
        s = checkpoint['samples']
    else:
        s = []
        checkpoint = None
    try:
        ip.load_state_dict(checkpoint['model_state_dict'])
        ip.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        last_epoch = checkpoint['epoch']

    except:
        last_epoch = 0
    del checkpoint

    if args.gpu:
        ip.to(torch.device("cuda"))
    print(f"Starting from {last_epoch}")
    acs = []
    mn = mp.Manager()
    results = mn.list([None]*args.samplesperbatch)
    states = mn.list()
    for i in range(args.loops):  # last_epoch, last_epoch+args.loops):
        try:
            m = manager(args.nenvs, args.seed, args.render, states)

            s += sample_self_play(ip, m)
            ds, state = get_random(s, args.samplesperbatch)
            print("tree", end=" ", flush=True)
            sm = start_monitoring(20, 1000)
            tm = TreeMan(ip, args.ntrees, state, results,
                         args.tree_iterations, args.render)
            while tm.remotes:
                tm.recv()
            sm.stop()

            tm.close()
            loss = train_policy(ds, list(results), ip)
            print(f"Loop {i} ended, {loss=}")
            del s[:int(len(s)-args.max_record)]
            save = {'samples': s}
            if not args.debug:
                save.update({
                    'epoch': i,
                    'model_state_dict': ip.state_dict(),
                    'optim_state_dict': ip.optimizer.state_dict(),
                })
            torch.save(save, args.path)
            gc.collect()
        except KeyboardInterrupt as e:
            if len(s):
                save = {'samples': s}
            if not args.debug and i > 0:
                save.update({
                    'epoch': i,
                    'model_state_dict': ip.state_dict(),
                    'optim_state_dict': ip.optimizer.state_dict(),
                })
            torch.save(save, args.path)
            raise e
