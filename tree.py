import numpy as np
import tetris
import weakref
import args

np.set_printoptions(precision=3)#,suppress=True)

class ExtendedRef(weakref.ref):
    def __init__(self, ob, callback=None):
        super().__init__(ob, callback)


    def __call__(self):
        """Return a pair containing the referent and the number of
        times the reference has been called.
        """
        ob = super().__call__()
        return ob
lose_rew=args.lose_rew
class Node:
    def __init__(self, state, x=None,y=None,parent=None, depth=0):
        if type(state) == tetris.Container:
            self.state = state
        else:
            self.state = tetris.Container(state)
            #raise TypeError(
            #    f"Not a tetris.Container: {state}")
        self.parent = parent
        self.children = np.tile(None, (10, 10))
        self.visits = 0
        self.depth = depth
        self.reward1 = 0
        self.reward2 = 0
        self.p1=None
        self.p2=None
        self.sum_r1 = np.zeros(10)
        self.sum_r2 = np.zeros(10)
        self.sum_p1 = np.tile(1,10)
        self.sum_p2 = np.tile(1,10)
        self.x=x
        self.y=y
        self.is_terminal=False


    def add_child(self, ac, child_state):
        if child_state == -1:
            self.children[ac] = False
        else:
            child = Node(child_state, x=ac[0],y=ac[1],
                         parent=ExtendedRef(self), depth=self.depth+1)
            self.children[ac] = child
            return child


    def update(self, reward1, reward2):
        # Increment the visits first to avoid division by 0
        self.visits += 1

        # Update rewards using a moving average for efficiency
        self.reward1 += (reward1 - self.reward1) / self.visits
        self.reward2 += (reward2 - self.reward2) / self.visits

        # Update parent's cumulative rewards and visits
        if self.parent:
            parent = self.parent()
            parent.sum_p1[self.x] += 1
            parent.sum_r1[self.x] += (reward1 - parent.sum_r1[self.x]) / parent.sum_p1[self.x]

            parent.sum_p2[self.y] += 1
            parent.sum_r2[self.y] += (reward2 - parent.sum_r2[self.y]) / parent.sum_p2[self.y]

    def backpropagate(self, reward1, reward2):
        self.update(reward1, reward2)
        if self.parent is None:
            return
        self.parent().backpropagate(reward1*0.99,reward2*0.99)
    def set_invalid(self,ac,ind):
        if ind==0:
            self.sum_p1[ac]=0
            self.children[ac]=False
        if ind==1:
            self.sum_p2[ac]=0
            self.children[:,ac]=False

        if np.all(self.children==False):
            self.is_terminal=True
    def expand(self,x,y):
        new_state = self.state.copy()
        new_state.step(x,y)
        s = new_state.get_state()
        valid=1
        if (s[0] == 125):
            valid=0
            self.set_invalid(x,0)
        if (s[232] == 125):
            valid=0
            self.set_invalid(y,1)
        if valid:
            child=self.add_child((x, y), new_state)
            filled=new_state.check_filled()
            r,is_terminal=calc_rew(s,(x,y),filled)
            child.is_terminal=is_terminal
            child.backpropagate(*r)
            return child

    def best_child(self, c_param=0.2):
        if self.visits == 0:
            raise ValueError

        log_visits = np.log(self.visits)

        # UCB for player 1
        valid_indices_p1 = np.where(self.sum_p1 > 0)[0]
        if valid_indices_p1.size > 0:
            sum_p1_valid = self.sum_p1[valid_indices_p1]
            sum_r1_valid = self.sum_r1[valid_indices_p1]
            ucb1 = (sum_r1_valid / sum_p1_valid) + c_param * np.sqrt(log_visits / sum_p1_valid)

            if self.p1 is not None:
                p1_valid = self.p1[valid_indices_p1]
                ucb1 += 100 * p1_valid / (sum_p1_valid + 1)

            best_move_p1 = valid_indices_p1[np.argmax(ucb1)]
        else:
            raise ValueError("No valid actions for player 1")

        # UCB for player 2
        valid_indices_p2 = np.where(self.sum_p2 > 0)[0]
        if valid_indices_p2.size > 0:
            sum_p2_valid = self.sum_p2[valid_indices_p2]
            sum_r2_valid = self.sum_r2[valid_indices_p2]
            ucb2 = (sum_r2_valid / sum_p2_valid) + c_param * np.sqrt(log_visits / sum_p2_valid)

            if self.p2 is not None:
                p2_valid = self.p2[valid_indices_p2]
                ucb2 += 100 * p2_valid / (sum_p2_valid + 1)

            best_move_p2 = valid_indices_p2[np.argmax(ucb2)]
        else:
            raise ValueError("No valid actions for player 2")

        return best_move_p1, best_move_p2
    def sumr(self):

        r1 = self.r1/np.maximum(1,self.p1)
        r2 = self.r2/np.maximum(1,self.p2)

        return r1,r2

    def collect_nodes(self):
        nodes = []
        #print(f"Depth: {depth}, Node: {self.depth}")
        for child in self.children.flatten():
            if child:
                nodes.extend(child.collect_nodes())
        nodes.append(self)
        return nodes

class MCTS:
    def __init__(self, i,render,data,c=0.2,results=None,lock=None):
        self.rendering = render and i==0
        self.data = np.frombuffer(data)[i*500:(i+1)*500]
        self.i=i
        self.lock=lock
        self.c=c
        self.results=results
        self.r=None
        self.remote=None
    def search(self, root, results=None,remote=None,budget=10000):

        if results:
            self.results=results
        self.remote = remote
        if type(root) == Node:
            self.root = root
            first=False
        else:
            self.root = Node(root)
            first=True

        for it in range(budget):
            leaf,acs = self.traverse(self.root)
            if not leaf.is_terminal:
                simulation_reward = self.rollout(leaf.state, acs)
                leaf.backpropagate(*simulation_reward)
            else:
                leaf.backpropagate(leaf.reward1,leaf.reward2)

        if self.results is not None:
            result =(self.root.sum_p1/self.root.visits,
                     self.root.sum_p2/self.root.visits)
            with self.lock:
                self.results.append((self.root.state.get_state(),result))
        #print(f"reward p1={self.root.reward1/self.root.visits:.3f} p2={self.root.reward2/self.root.visits:.3f}")
        if (first and self.remote):
            self.finalize(budget//10)
        del self.root

    def traverse(self, node,d=0):
        init_count=node.depth
        while np.any(node.children) and not node.is_terminal:
            acs=node.best_child(self.c)
            if node.children[acs]==None:
                new_node= node.expand(*acs)
                if new_node:
                    return new_node,acs

            else:
                node = node.children[acs]

        if node.is_terminal or np.all(node.children == False):
            node.is_terminal=True
            return node,acs
        while not node.is_terminal:
            if self.remote:
                state=node.state.get_state()
                self.data[:]=state
                self.remote.send(self.i)
                policy = self.remote.recv()[0]
                node.p1=policy[0]
                node.p2=policy[1]
                acs=node.best_children(self.c)
            else:
                arr=np.array(node.children!=False,dtype=bool)
                true_indices = np.argwhere(arr)
                acs = true_indices[np.random.randint(len(true_indices))]
            new_node=node.expand(*acs)
            if new_node:
                return new_node,acs

            #if np.all(arr) and np.any(node.children) and not node.is_terminal:
            #    return self.traverse(node,d=1)
        return node,acs
    def rollout(self, node, actions):
        is_terminal = False
        state = node.copy()
        reward = np.zeros(2)
        depth=0
        while not is_terminal:
            x = state.get_state()

            # Calculate rewards and terminal status
            filled = state.check_filled()
            reward_increment, is_terminal = calc_rew(x, actions, filled)
            reward += reward_increment*0.99**depth

            # If terminal, exit loop
            if is_terminal:
                break

            depth+=1
            # Choose actions
            if self.remote:
                self.data[:] = x
                self.remote.send((self.i))
                prob = self.remote.recv()[0]
                actions = np.random.choice(10, p=prob[0]), np.random.choice(10, p=prob[1])
            else:
                actions = np.random.randint(10, size=2)

            state.step(*actions)
            self.render(state)

        return reward
    def render(self,state):
        if self.r is None:
            self.r =tetris.Renderer(1,10) if self.rendering else False
        if self.r:
            try:
                r = self.r.render(state)
            except:
                print("render error")
                r=1
            if r:
                #self.rendering = False
                self.r.close()
                self.r=False
    def finalize(self,budget=None):
        if not budget:
            return
        keep=[]
        nodes = self.root.collect_nodes()
        nodes.remove(self.root)
        all_rew = np.array([node.sumr() if np.any(node.children) else np.tile(np.nan,(2,10)) for node in nodes])
        not_all_nan=np.where(np.count_nonzero(~np.isnan(all_rew),(1,2))>2)
        if not np.any(not_all_nan):
            #print("nothing to keep")
            return

        rews=all_rew[not_all_nan]
        #nrews=rews
        #print(rews.std(-1))
        #nadv=adv/adv.std(-1,keepdims=True)
        mask = np.nanmax(np.nanstd(rews,-1),-1) #np.max(nadv, axis=(1,2)) > 2
        #print(f"max: {np.nanmax(mask):.2f}")

        keep.extend(np.array(nodes)[not_all_nan][mask>1.5])
        adv=(rews-np.nanmean(rews,axis=-1,keepdims=True))[mask>1.5]


        if not keep:
            #print(f"nothing to keep, max: {np.nanmax(mask):.2f}")
            return
            #raise Exception("filter too high")
        self.remote.send([k.state.get_state() for k in keep])
        p1,p2=self.remote.recv()
        interest=np.nanmax(np.array((np.nansum(adv[:,0]*p1,-1),np.nansum(adv[:,1]*p2,-1))),0)
        #print(f"{interest=}")
        selection=np.array(keep)[interest>3.5]
        #print(f"{len(selection)} interesting states")
        del keep
        for sel in selection:
            sel.parent=None
            self.search(sel,results=self.results,remote=self.remote,budget=budget)

def calc_rew(x,actions,filled):

    lose = (x[0] == 127)
    win = (x[0] == 126)
    is_terminal = win or lose
    if is_terminal:
        reward = np.array([10., lose_rew] if win else [lose_rew, 10.]) / 10
        if x[232] == 127:
            reward[1] = lose_rew
        return reward, is_terminal
    reward=np.clip(filled, 0, 0.7) * 0.1 - 0.02
    for i, (r,g) in enumerate(((x[0], x[5]),(x[232],x[237]))):
        if r < 125:
            lines = (r % 15)//3
            atk = r//15
            # hard drop without breaking b2b or spin
            reward[i] += (r % 3)*0.01+(0.09 if x[i*232+8] == 5 and actions[i] !=1 else 0) #(0.01 if actions[i] ==1
                         # else (0.2 if x[i*232+8] == 5 else 0.001))*
            # t spin, regular, tetris
            reward[i] += ((1 if atk >= lines*2+1 and lines <4
                           else 0.3 if lines < 4 else 0.5)*(lines+1.5*atk)**1.2)/5
            # garbage
            reward[i]+=g*0.5
    return reward,is_terminal

