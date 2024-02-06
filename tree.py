import numpy as np
import sim


class Node:
    def __init__(self, state, parent=None):

        if type(state) == tuple:
            self.state = sim.Container(*state)
        else:
            self.state = state
        self.parent = parent
        self.children = np.tile(None, (10, 10))
        self.visits = 0
        self.reward1 = 0
        self.reward2 = 0
        self.is_terminal = False

    def add_child(self, ac, child_state):
        if child_state == -1:
            self.children[ac] = False
        else:
            child = Node(child_state, parent=self)
            self.children[ac] = child

    def update(self, reward1, reward2):
        self.visits += 1
        self.reward1 += reward1
        self.reward2 += reward2

    def not_expanded(self):
        return np.all(None in self.children)

    def best_child(self, c_param=0.25):
        r, v = self.sum_rv(self.children, 1)
        if self.visits == 0:
            return (1, 1)
        ac1 = np.nanargmax([[(rew / vis) + c_param * np.sqrt((2 * np.log(self.visits) / vis)) if v else False
                             ] for rew, vis in zip(r, v)])
        r, v = self.sum_rv(self.children.T, 2)
        ac2 = np.nanargmax([[(rew / vis) + c_param * np.sqrt((2 * np.log(self.visits) / vis)) if v else False
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

    def search(self, root, results, i, remote, budget=10000):
        self.remote = remote
        self.root = Node(root)
        self.i = i
        it = 0
        if self.render:
            self.r = sim.Renderer(1, 20)
        while it < budget:
            leaf = self.traverse(self.root)  # phase 1
            if not leaf:
                continue
            it += 1
            simulation_reward = self.rollout(leaf.state)
            self.backpropagate(leaf, *simulation_reward)  # phase 3
        remote.send((None, None, None))
        results[i] = ([(rew/vis if rew else -0.2) for rew, vis in zip(*Node.sum_rv(self.root.children, 1))],
                      [(rew/vis if rew else -0.2) for rew, vis in zip(*Node.sum_rv(self.root.children.T, 2))])

    def traverse(self, node):
        while not node.not_expanded():
            n = node.children[node.best_child()]
            if n:
                node = n
        if node.is_terminal:
            return node
        else:
            x, y = np.where(node.children == None)

            new_state = node.state.copy()
            new_state.step(x[0], y[0])
            s = new_state.get_state()
            if s[0] == 125 or s[738] == 125:
                node.add_child((x[0], y[0]), -1)
            else:
                node.add_child((x[0], y[0]), new_state)
            return node.children[x[0], y[0]]

    def rollout(self, state):
        is_terminal = False
        while not is_terminal:

            x = state.get_state()
            y = state.get_atk()
            lose = (x[0] == 127 or (y[0] and sum(y[1]) >= 30))
            win = (x[0] == 126 or (not y[0] and sum(y[1]) >= 30))
            is_terminal = win or lose
            self.remote.send((x, y, self.i))
            action1, action2 = self.remote.recv()
            state.step(action1, action2)

            if self.render:
                self.r.render(state)

        self.end_reward = [1, -0.5] if win else [-0.5,
                                                 1] if x[738] == 126 else [-0.5, -0.5]

        return self.end_reward

    def backpropagate(self, node, reward1, reward2):
        while node is not None:
            node.update(reward1, reward2)
            node = node.parent
