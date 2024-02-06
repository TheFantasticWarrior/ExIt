
import numpy as np
import multiprocessing as mp
import sim
from multiprocessing import Pipe, Process
from tree import MCTS
from torch import no_grad


class manager:
    def __init__(self, nenvs, seed, render, states) -> None:
        cpus = mp.cpu_count()
        try:
            assert nenvs % cpus == 0 or nenvs < cpus and nenvs % 2 == 0
            self.nenvs = nenvs//2
        except:
            print("setting nenvs to 1")
            self.nenvs = 1
        self.nremotes = min(cpus, self.nenvs)
        self.seeds = np.arange(self.nenvs)+seed
        # self.envs=[make_env(seed+i) for i in range(nenvs)]
        env_seeds = np.array_split(self.seeds, self.nremotes)
        self.q = states
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=gather_data, args=(work_remote, remote, seeds, self.q, seeds == seed and render))
                   for work_remote, remote, seeds in zip(self.work_remotes, self.remotes, env_seeds)]
        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def send(self, actions):
        actions = actions.numpy()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def recv(self):
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        a, b, c = zip(*results)

        return _flatten_obs(a).reshape(-1, 1476), b, c

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
    def __init__(self, seed, q, render=False):
        x = sim.Container()
        x.seed_reset(seed)
        self.states = []
        self.x = x
        if render:
            self.r = sim.Renderer(1, 20)
            self.rt = True
        else:
            self.rt = False
        self.q = q
        self.a = 0

    def no_step(self):
        x = self.x.get_state()
        return x, self.x.get_atk(), [x[0], x[738]]

    def step_both(self, a):
        self.x.step(a[0], a[1])
        x = self.x.get_state()
        atk = self.x.get_atk()
        reward = [0, 0]
        max_spike = sum(atk[1]) >= 30
        lose = x[0] == 127 or (atk[0] and max_spike)
        win = x[0] == 126 or (max_spike and not atk[0])
        self.a += 1
        is_terminal = self.a >= 10  # win or lose
        if is_terminal:
            self.a = 0

            self.end_reward = [1, -0.5] if win else [-0.5,
                                                     1] if x[738] == 126 else [-0.5, -0.5]

            state = self.states[np.random.randint(len(self.states))]
            self.states = []
            self.x.reset()
            x = self.x.get_state()
            atk = self.x.get_atk()
            self.q.append(state)

            return x, atk, (1, self.end_reward)
        else:
            b, c = self.x.get_hidden()
            self.states.append(
                ((x, atk), (np.concatenate([x[:222], x[738:960]]), b, c, len(atk[1]), atk[1])))

            for i, r in enumerate((x[0], x[738])):
                if r != 125:
                    lines = (r % 10)//2
                    satk = r//10
                    reward[i] += (0.01 if a[i] ==
                                  1 else (0.1 if x[i*738+5] == 5 else 0.001))*(r % 2)
                    reward[i] += ((0.5 if satk >= lines*2+2 and lines <
                                  4 else 0.3 if lines < 4 else 0.25)*(lines+1.5*satk)**1.2)/20
                else:
                    reward[i] = -0.2

            return x, atk, reward

    def render_toggle(self):
        if self.rt:
            self.rt = False
            self.r.close()
            self.r = None
        else:
            self.r = sim.Renderer(1, 20)
            self.rt = True

    def render(self):
        self.r.render(self.x)

    def set_state(self, x):
        if type(x) == tuple:
            self.x = sim.Container(x)
        else:
            self.x = x.copy()
        return self.x.get_state(), self.x.get_atk()

    def close(self):
        if self.rt:
            self.r.close()


def gather_data(remote, parent_remote, seeds, q, render):
    parent_remote.close()
    envs = [envc(seed, q, i == 0 and render) for i, seed in enumerate(seeds)]
    remote.send([env.no_step() for env in envs])
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                actions = data
                actions = actions.reshape(-1, 2)
                data = [env.step_both(action)
                        for env, action in zip(envs, actions)]
                remote.send(data)

                if envs[0].rt:
                    envs[0].render()

            elif cmd == "to_state":
                state, i = data

                remote.send(envs[i].set_state(state))

            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                envs[0].render_toggle()
            elif cmd == 'close':
                for env in envs:
                    env.close()
                remote.close()
                break

    except KeyboardInterrupt:
        pass
    finally:
        for env in envs:
            env.close()


class TreeMan:
    def __init__(self, nn, nenvs, to_process, results, budget, render=False) -> None:
        cpus = mp.cpu_count()
        self.nn = nn

        try:
            assert nenvs % cpus == 0 or nenvs < cpus and nenvs % 2 == 0
            self.nenvs = nenvs//2
        except:
            print("setting nenvs to 1")
            self.nenvs = 1
        self.nremotes = min(cpus, self.nenvs)
        self.trees = [(MCTS(i == 0 and render), i)
                      for i in range(self.nremotes)]
        self.states = list([None]*self.nremotes)
        self.atk = list([None]*self.nremotes)
        self.tremotes = list([None]*self.nremotes)
        env_trees = self.trees  # np.array_split(self.trees, self.nremotes)
        # np.array_split(states, self.nremotes)
        self.q = results
        self.loc = list()
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.nremotes)])
        self.remotes = list(self.remotes)
        self.ps = [Process(target=tree_gather, args=(work_remote, remote, tree, self.q, states, budget))
                   for work_remote, remote, tree, states in zip(self.work_remotes, self.remotes, env_trees, to_process)]
        for p in self.ps:
            p.start()
        self.to_process = to_process[self.nremotes:]
        for remote in self.work_remotes:
            remote.close()

    def recv(self):
        states, atk, _ = map(list, zip(
            *[remote.recv()
              for remote in self.remotes]))
        x = len(states)-1
        for i, s in enumerate(reversed(states)):
            if s is None:
                if len(self.to_process):
                    self.remotes[x-i].send(self.to_process[0])
                    self.to_process = self.to_process[1:]
                else:
                    del atk[x-i]
                    del states[x-i]
                    self.remotes[x-i].send(None)
                    self.remotes[x-i].close()
                    del self.remotes[x-i]

        if len(atk):
            with no_grad():
                acs = self.nn(states, atk, p2=True)
                acs = acs.cpu().reshape(-1, 2)
            for r, a in zip(self.remotes, acs):
                r.send(a)

    def close(self):
        for p in self.ps:
            p.join()


def tree_gather(remote, parent_remote, trees, q, states, budget):
    parent_remote.close()
    try:
        while True:
            trees[0].search(states, q, trees[1], remote, budget)
            states = remote.recv()
            if not states:
                break
    except KeyboardInterrupt:
        pass


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
