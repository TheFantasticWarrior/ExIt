from core import initial_policy,random_play, sample_self_play, get_random, train_policy
import args
from mp import manager, TreeMan
import torch.multiprocessing as mp
import torch
import gc
#from hanging_threads import start_monitoring

if __name__ == "__main__":
    ip = initial_policy()
    if args.load:
        checkpoint = torch.load(args.path+"save")
        d = checkpoint['dataset']
    else:
        d = [[],[]]
        checkpoint = None
        last_epoch = 0
        avgs=[]
    if checkpoint:
        try:
            #raise
            ip.load_state_dict(checkpoint['model_state_dict'])
            ip.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            last_epoch = checkpoint['epoch']
            avgs=checkpoint['avgs']
        except:
            last_epoch = 0
            avgs=[]
    del checkpoint

    if args.gpu:
        ip.to(torch.device("cuda"))
    print(f"Starting from {last_epoch}")
    acs = []
    mn = mp.Manager()
    lock=mn.Lock()
    results = mn.list()
    if not args.load:
        m = manager(args.nenvs, args.seed, args.render)
        state=random_play(m)
        s=state
        try:
            tm = TreeMan(None, args.ntrees, state, results,args.tree_iterations//10,lock, args.render)
            tm.run()
            state, acs = zip(*results)
            del results[:]
            d[0].extend(state)
            d[1].extend(acs)
            loss = train_policy(*d,ip)
            print(f"Init ended, {loss=}")
        except:
            print("init failed")
            torch.save({'dataset':d}, args.path+"save")
            raise
    for it in range(last_epoch, last_epoch+args.loops):
        try:
            m = manager(args.nenvs, args.seed, args.render)

            if not(args.load and it==0 and len(d)>=args.samplesperbatch):
                state = sample_self_play(ip, m)
                avg=m.lines/m.count if m.count else 0
                print(f"average lines: {avg:.3f}")
                avgs.append(avg)
            #sm = start_monitoring(20, 1000)
            tm = TreeMan(ip, args.ntrees, state, results,args.tree_iterations,lock, args.render and it%1==0)
            tm.run()
            with lock:
                state, acs = zip(*results)
                del results[:]
            d[0].extend(state)
            d[1].extend(acs)

            ds=get_random(d,args.samplesperbatch)
            loss = train_policy(*ds,ip)
            print(f"Loop {it} ended, {loss=}")
            for i in range(2):
                del d[i][:int(len(d[i])-args.max_record)]
            save = {'dataset': d}
            if not args.debug:
                save.update({
                    'epoch': it,
                    'avgs': avgs,
                    'model_state_dict': ip.state_dict(),
                    'optim_state_dict': ip.optimizer.state_dict(),
                })
            torch.save(save, args.path+"save")
        except:
            if len(d):
                save = {'dataset': d}
                if not args.debug and it > 0:
                    save.update({
                        'epoch': it,
                        'avgs':avgs,
                        'model_state_dict': ip.state_dict(),
                        'optim_state_dict': ip.optimizer.state_dict(),
                    })
                torch.save(save, args.path+"errorsave")
            raise
