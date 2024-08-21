from core import initial_policy,random_play, sample_self_play
import args
from mp import manager, TreeMan
import torch.multiprocessing as mp
import torch
from torch.utils.data import random_split,DataLoader,Dataset
import time
import numpy as np
class Dataset(Dataset):
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.gpu_states=[]
        self.gpu=False
    def __len__(self):
        if self.gpu:
            return len(self.gpu_states)
        return len(self.states)
    def __getitem__(self, idx):
        if self.gpu:
            return self.gpu_states[idx], self.gpu_actions[idx]
        return self.states[idx], self.actions[idx]
    def add(self,states,actions):
        self.states.extend(states)
        self.actions.extend(actions)
    def to_gpu(self,l=None):
        s=slice(l)
        self.gpu_states=torch.tensor(np.array(self.states[s]), dtype=torch.int,device=torch.device("cuda"))
        self.gpu_actions=torch.tensor(np.array(self.actions[s]),device=torch.device("cuda"))+1e-7
        self.gpu=True


def train_epoch(data, model,val):
    loss=[]
    for X,y in data:
        neglogp = model(X, p2=True, loss=True)
        #y = y.to(torch.device("cuda")) #(zip(*(d for d in y if d is not None)))
        loss.append(model.backward(neglogp,y,val))

    return np.mean(loss)

def train_loop(train,test,model):
    train.to_gpu()
    test.to_gpu()
    train_loader = DataLoader(train, batch_size=512,shuffle=True)
    test_loader = DataLoader(test, batch_size=512)
    train_loss=5
    old_train_loss=5
    val_loss=5
    old_val_loss=5
    i=0
    while True:
        # Training phase
        t=time.time()
        train_loss = train_epoch(train_loader, model, False)
        nt=time.time()
        print(f"Train loss: {train_loss:.7f}, time: {nt-t:.2f}, lr: {model.lr.get_last_lr()[0]:.5f}")

        # Validation phase
        with torch.no_grad():
            val_loss = train_epoch(test_loader, model, True)
        print(f"Val loss: {val_loss:.7f}, time:{time.time()-nt:.2f}, decrease: {old_val_loss-val_loss}")

        # Check for overfitting
        if train_loss < old_train_loss and val_loss > old_val_loss and val_loss>train_loss:
            i += 1
            if i == 3:
                print("Overfitting detected, early stopping.")
                break
        else:
            i = 0  # Reset counter if there's no overfitting
            old_val_loss = min(val_loss,old_val_loss)
            old_train_loss = min(train_loss,old_train_loss)
if __name__ == "__main__":
    ip = initial_policy()
    train=Dataset()
    test=Dataset()

    last_epoch = 0
    avgs=[]
    checkpoint = None
    if args.load:
        if not args.reset_dataset:
            checkpoint = torch.load(args.path)
            train=checkpoint["train"]
            test=checkpoint["test"]
            try:
                last_epoch = checkpoint['epoch']
                avgs=checkpoint['avgs']
            except:
                pass
        try:
            #raise
            ip.load()
        except:
            print("Can't load model,continuing without save data.")
            if len(train):
                print("found existing dataset, training")
                train_loop(train,test,ip)


    del checkpoint

    print(f"Starting from {last_epoch}")
    acs = []
    mn = mp.Manager()
    lock=mn.Lock()
    results = mn.list()
    if not len(train):
        m = manager(args.nenvs, args.seed, args.render)
        state=random_play(m)
        s=state
        try:
            tm = TreeMan(None, args.ntrees, state, results,0.2,lock, args.render)
            tm.run()
            d=Dataset()
            d.add(*zip(*results))
            split_idx = int(0.9 * len(d))
            train.add(d.states[:split_idx],d.actions[:split_idx])
            test.add(d.states[split_idx:],d.actions[split_idx:])
            train_loop(train,test,ip)
            print(f"Init ended")
            torch.save({'train':train,'test':test}, args.path)
            ip.save()
        except Exception:
            import traceback
            traceback.print_exc()
            print("init failed")
            breakpoint()
            torch.save({'train':train,
                    'test':test}, args.path+"error")
            raise

    #for it in range(min(last_epoch,1), last_epoch+args.loops):
    it=last_epoch
    while len(train)<20000:
        it+=1
        try:
            m = manager(args.nenvs, args.seed+it*1200, args.render)

            state = sample_self_play(ip, m,1000)
            avg=float(m.lines/m.count) if m.count else 0
            avgs.append(avg)
            print(f"{avg=}")
            tm = TreeMan(None, args.ntrees, state, results,0.05,lock, args.render and it%1==0)
            tm.run()
            if not len(results):
                raise ValueError("no results")
            with lock:
                try:
                    d=Dataset()
                    d.add(*zip(*results))
                    split_idx = int(0.9 * len(d))
                    train.add(d.states[:split_idx],d.actions[:split_idx])
                    test.add(d.states[split_idx:],d.actions[split_idx:])
                except:
                    breakpoint()
                del results[:]
            train_loop(train,test,ip)
            print(f"Loop {it} ended")
            save = {'train':train,
                    'test':test}
            if not args.debug:
                save.update({
                    'epoch': it+1,
                    'avgs': avgs,
                })
            torch.save(save, args.path)
            ip.save()
        except:
            if len(train):
                save = {'train':train,
                    'test':test}

                if not args.debug and it > 0:
                    save.update({
                        'epoch': it,
                        'avgs':avgs
                    })
                torch.save(save, args.path+"error")
            ip.save("error_model")
            raise
