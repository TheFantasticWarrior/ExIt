import numpy as np
import sim
import weakref


class Node:
    def __init__(self, state, parent=None, depth=0):

        if type(state) == tuple:
            self.state = sim.Container(*state)
        elif type(state) == sim.Container:
            self.state = state
        else:
            raise TypeError(
                f"Not a game container or valid creator tuple: {state}")
        self.parent = parent
        self.children = np.tile(None, (10, 10))
        self.visits = 0
        self.depth = depth
        self.reward1 = 0
        self.reward2 = 0
        self.is_terminal = False

    def add_child(self, ac, child_state):
        if child_state == -1:
            self.children[ac] = False
        else:
            child = Node(child_state, parent=weakref.ref(
                self), depth=self.depth+1)
            self.children[ac] = child

    def update(self, reward1, reward2):
        self.visits += 1
        self.reward1 += reward1
        self.reward2 += reward2

    def not_expanded(self):
        return np.all(None in self.children)

    def best_child(self, c_param=0.25):
        if self.visits == 0:
            return (1, 1)
        r, v = self.sum_rv(self.children, 1)
        ac1 = np.nanargmax([[(rew / vis) + c_param * np.sqrt((2 * np.log(self.visits) / vis)) if vis is not np.nan else 0
                             ] for rew, vis in zip(r, v)])
        r, v = self.sum_rv(self.children.T, 2)
        ac2 = np.nanargmax([[(rew / vis) + c_param * np.sqrt((2 * np.log(self.visits) / vis)) if vis is not np.nan else 0
                             ] for rew, vis in zip(r, v)])
        return (ac1, ac2)

    @staticmethod
    def sum_rv(arr, a):
        r = [sum([getattr(c, f"reward{a}") for c in x if c])
             if x.any() else np.nan for x in arr]
        v = [sum([c.visits for c in x if c])
             if x.any() else np.nan for x in arr]
        return r, v


class MCTS:
    def __init__(self, render=False):
        self.render = render
        self.r = None

    def search(self, root, results=None, i=None, remote=None, budget=10000):

        self.remote = remote
        self.root = Node(root)
        self.i = i

        if self.render and not self.r:
            self.r = sim.Renderer(1, 20)

        for _ in range(budget):
            leaf = self.traverse(self.root)  # phase 1
            if np.all(leaf.children == False):
                self.backpropagate(leaf, -1, -1)
                continue
            simulation_reward = self.rollout(leaf.state, leaf.depth)
            self.backpropagate(leaf, *simulation_reward)  # phase 3
        if results is None:
            return
        remote.send((None, None))
        result = ([(rew/vis if rew else 0) for rew, vis in zip(*Node.sum_rv(self.root.children, 1))],
                  [(rew/vis if rew else 0) for rew, vis in zip(*Node.sum_rv(self.root.children.T, 2))])
        result = tuple(map(lambda x: [el/sum(x) for el in x], result))
        results[i] = result
        del self.root

    def traverse(self, node):
        while not node.not_expanded() and not np.all(node.children == False):
            n = node.children[node.best_child()]
            if n:
                node = n
        if node.is_terminal or np.all(node.children == False):
            return node
        else:
            new_state = node.state.copy()

            while True:
                x, y = np.where(node.children == None)
                if not len(x):
                    del new_state
                    return self.traverse(node)
                new_state.step(x[0], y[0])
                s = new_state.get_state()
                if s[0] == 125 or s[738] == 125 and x[0] != 1 and y[0] != 1:
                    node.add_child((x[0], y[0]), -1)
                else:
                    node.add_child((x[0], y[0]), new_state)
                    del new_state
                    return node.children[x[0], y[0]]

    def rollout(self, state, depth):
        is_terminal = False
        reward = np.array([0., 0.])

        while not is_terminal:
            depth += 1
            x = state.get_state()
            y = state.get_atk()
            lose = (x[0] == 127 or (y[0] and sum(y[1]) >= 30))
            win = (x[0] == 126 or (not y[0] and sum(y[1]) >= 30))
            is_terminal = win or lose
            if not is_terminal:
                self.remote.send((x, y))
                actions = self.remote.recv()
                state.step(*actions)

                if self.render:
                    r = self.r.render(state)
                    if r:
                        self.render = False
                        self.r.close()
                        del self.r
                for i, r in enumerate((x[0], x[738])):
                    if r != 125:
                        lines = (r % 10)//2
                        satk = r//10
                        reward[i] += (0.01 if actions[i] ==
                                      1 else (0.1 if x[i*738+5] == 5 else 0.001))*(r % 2)
                        reward[i] += ((0.5 if satk >= lines*2+2 and lines <
                                       4 else 0.3 if lines < 4 else 0.25)*(lines+1.5*satk)**1.2)/20
                    else:
                        reward[i] += -0.2

        reward += [1., -0.5] if win else [-0.5,
                                          1.] if x[738] == 126 else [-0.5, -0.5]

        return map(lambda x: x/depth, reward)

    def backpropagate(self, node, reward1, reward2):
        while node is not None:
            node.update(reward1, reward2)
            if node.parent is None:
                return
            node = node.parent()
