
import numpy as np
import torch.multiprocessing as mp
from multiprocessing.connection import wait
import tetris
from torch.multiprocessing import Pipe, Process,Manager,Array
from tree import MCTS
from torch import no_grad
import args


class manager:
    def __init__(self, nenvs, seed, render) -> None:
        cpus = mp.cpu_count()
        try:
            assert nenvs % cpus == 0 or nenvs < cpus
            self.nenvs = nenvs
        except:
            print("setting nenvs to 1")
            self.nenvs = 1
        self.nremotes = min(cpus, self.nenvs)
        self.seeds = np.arange(self.nenvs)
        # self.envs=[make_env(seed+i) for i in range(nenvs)]
        env_seeds = np.array_split(self.seeds, self.nremotes)
        self.mn=Manager()
        self.q = self.mn.list()
        self.lock=self.mn.Lock()
        self.data=Array("d",500*self.nenvs,lock=False)
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=gather_data, args=(work_remote, remote, seeds, self.q, seeds == 0 & render,self.lock,self.data,seed))
                   for work_remote, remote, seeds in zip(self.work_remotes, self.remotes, env_seeds)]
        for p in self.ps:
            p.daemon = True
            p.start()
        self.count=0
        self.lines=0
        for remote in self.work_remotes:
            remote.close()

    def send(self, actions):
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def recv(self):
        results = [remote.recv() for remote in self.remotes]
        if results[0] is None:
            return
        c = _flatten_list(results)

        c=np.stack(c)
        self.count+=c.size - np.count_nonzero(np.isnan(c))
        self.lines+=np.nansum(c) if self.count else 0

    def render_toggle(self):
        self.remotes[0].send(("render", None))

    def reset(self):

        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs).reshape(-1, 1476)

    def to_state(self, x):
        for remote in self.remotes:
            remote.send(('to_state', x))
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        a, b = zip(*results)
        return _flatten_obs(a).reshape(-1, 1476), b

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

class envc:
    def __init__(self, seed,i,shared_arr, q,lock, render=False):
        self.share = np.frombuffer(shared_arr)[i*500:(i+1)*500]
        x = tetris.Container()
        x.seed_reset(seed)
        self.states = []
        self.x = x
        if render:
            self.r = tetris.Renderer(1, 20)
            self.rt = True
        else:
            self.rt = False
        self.q = q
        self.total_lines=np.array([0,0])
        self.lock=lock


    def no_step(self):
        self.share[:] = self.x.get_state()
    def step_both(self, a):
        self.x.step(a[0], a[1])
        x = self.x.get_state()
        reward =np.array([0, 0])

        lose = x[0] == 127 or x[464]<-30
        win = x[0] == 126 or x[464]>30
        is_terminal = win or lose
        if is_terminal:

            #end_reward = [1, -0.5] if win else [-0.5, 1] if x[738] == 126 else [-0.5, -0.5]
            #self.total_reward+=end_reward
            state = self.states[np.random.randint(len(self.states))]
            self.states = []
            self.x.reset()
            self.share[:] = self.x.get_state()
            with self.lock:
                self.q.append(state)

            return self.total_lines
        else:
            self.share[:]=x
            self.states.append(self.x.copy())
            for i in range(2):
                r=x[232*i]
                self.total_lines[i]+=(r % 10)//2 if r!=125 else 0
            """
            for i, (r,g) in enumerate(((x[0], x[232]),(x[5],x[237]))):
                if r != 125:
                    lines = (r % 10)//2
                    #satk = r//10
                    # hard drop without breaking b2b or spin
                    #reward[i] += (0.15 if a[i] == 1 else
                                  (0.2 if x[i*232+8] == 5 else 0.1))*(r % 2)
                    # t spin, regular, tetris
                    #reward[i] += ((1 if satk >= lines*2+2 and lines <4
                                   else 0.3 if lines < 4 else 0.5)*
                                  (lines+1.5*satk)**1.2)/5
                    # garbage
                    #reward[i]+=g*0.5

                    self.total_lines[i]+=lines
                #else:
                    #reward[i] = -0.2
            """
            return np.array([np.nan,np.nan])

    def render_toggle(self):
        if self.rt:
            self.rt = False
            self.r.close()
            self.r = None
        else:
            self.r = tetris.Renderer(1, 20)
            self.rt = True

    def render(self):
        r = self.r.render(self.x)
        if r:
            self.render_toggle()

    def set_state(self, x):
        if type(x) == tuple:
            self.x = tetris.Container(x)
        else:
            self.x = x.copy()
        return self.x.get_state()

    def close(self):
        if self.rt:
            self.r.close()


def gather_data(remote, parent_remote, i, q, render,lock,data,seed):
    parent_remote.close()
    envs = [envc(seed,order,data, q, lock,r) for r, order in zip(render, i)]
    for env in envs:
        env.no_step()
    remote.send(None)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                actions = data
                actions = actions.reshape(-1, 2)
                remote.send([env.step_both(action)
                for env, action in zip(envs, actions)])

                if envs[0].rt:
                    envs[0].render()
            elif cmd == 'reset':
                for env in envs:
                    env.reset()
                remote.send(None)
            elif cmd == 'render':
                envs[0].render_toggle()
            elif cmd == 'close':
                for env in envs:
                    env.close()
                remote.close()
                break
            """
            elif cmd == "to_state":
                state, i = data

                remote.send(envs[i].set_state(state))
            """

    except KeyboardInterrupt:
        pass
    except:
        raise
    finally:
        for env in envs:
            env.close()


class TreeMan:
    def __init__(self, nn, nenvs, to_process, results,budget,lock,render=False) -> None:
        cpus = mp.cpu_count()
        self.nn = nn
        try:
            assert nenvs % cpus == 0 or nenvs < cpus
            self.nenvs = nenvs
        except:
            print("setting nenvs to 1")
            self.nenvs = 1
        self.nremotes = min(self.nenvs, len(to_process))
        print(self.nremotes,"workers")
        self.data=Array("d",500*self.nremotes,lock=False)
        self.trees = [MCTS(i,render,self.data,results,lock)
                      for i in range(self.nremotes)]
        env_trees = self.trees  # np.array_split(self.trees, self.nremotes)
        # np.array_split(states, self.nremotes)
        self.q =results
        self.remotes, self.work_remotes = zip(
        *[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=tree_gather, args=(work_remote, remote, tree,states,self.q, budget,self.nn==None))
            for work_remote, remote, tree, states in zip(self.work_remotes, self.remotes, env_trees, to_process)]
        for p in self.ps:
            p.start()
        self.to_process = to_process[self.nremotes:]
        self.remotes = list(self.remotes)
        for remote in self.work_remotes:
            remote.close()
        #print("inited", end=" ", flush=True)
    def run(self):
        while self.remotes:
            self.recv()
        self.close()
    def recv(self):
        ready_list = wait(self.remotes, 0.1)
        if not ready_list:
            return
        try:
            x=zip(*[remote.recv() for remote in ready_list])
        except:
            self.remotes=[]
            return
        if x is None:
            return
        states,masks =map(list,x)
        """
        normal: index of tree,mask
        finalize: list of states,None
        done: None,None
        """
        finalize_indices = [i for i, mask in enumerate(masks) if mask is None]
        process_indices = [i for i, mask in enumerate(masks) if mask is not None]

        if process_indices:
            states_to_process = [states[i] for i in process_indices]
            masks_to_process = [masks[i] for i in process_indices]
            remotes_to_process = [ready_list[i] for i in process_indices]

            mask_batch = np.stack(masks_to_process, axis=0)
            state_batch = [self.data[s*500:(s+1)*500] for s in states_to_process]

            with no_grad():
                acs = self.nn(state_batch, p2=True, mask=mask_batch)
                acs = acs.detach()
                if args.gpu:
                    acs = acs.cpu()
                acs = acs.numpy().reshape(-1, 2)

            for remote, ac in zip(remotes_to_process, acs):
                remote.send(ac)
        # Process finalized states
        for i in finalize_indices:
            if isinstance(states[i], list):
                with no_grad():
                    result = self.nn(states[i], p2=True, loss=True)
                    ready_list[i].send(tuple(map(lambda x: x.cpu().numpy(), result)))
            else:
                if self.to_process:#args.samplesperbatch > len(self.q) + 1 and
                    ready_list[i].send(self.to_process.pop(0))
                else:
                    ready_list[i].send(None)
                    ready_list[i].close()
                    idx=self.remotes.index(ready_list[i])
                    del self.remotes[idx]
                    self.ps[idx].join()
                    del self.ps[idx]
            print(f"{len(self.q)}/{args.samplesperbatch}", end=" ", flush=True)

    def close(self):
        for p in self.ps:
            p.join()
        print()

def tree_gather(remote, parent_remote, tree,states, q, budget,rand):
    parent_remote.close()
    while True:
        if rand:
            tree.search(states, q, None, budget)
        else:
            tree.search(states, q, remote, budget)
        remote.send((None,None))
        states = remote.recv()

        if not states:
            remote.close()
            del tree
            break


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    try:
        assert isinstance(l, (list, tuple))
        assert len(l) > 0
        assert all([len(l_) > 0 for l_ in l])
    except Exception as e:
        [print(len(l_), l_) for l_ in l]
        raise e
    return [l__ for l_ in l for l__ in l_]
