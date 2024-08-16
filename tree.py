import numpy as np
import tetris
import weakref
import args

np.set_printoptions(precision=3)#,suppress=True)

lose_rew=args.lose_rew
class Node:
    def __init__(self, state, parent=None, depth=0):
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
        self.is_terminal=False


    def add_child(self, ac, child_state):
        if child_state == -1:
            self.children[ac] = False
        else:
            child = Node(child_state, parent=weakref.ref(
                self), depth=self.depth+1)
            self.children[ac] = child

    def update(self, reward1, reward2):
        self.reward1 = self.reward1/max(self.visits,1)+reward1
        self.reward2 = self.reward2/max(self.visits,1)+reward2
        self.visits += 1
    def backpropagate(self, reward1, reward2):
        self.update(reward1, reward2)
        if self.parent is None:
            return
        self.parent().backpropagate(reward1*0.99,reward2*0.99)
    def set_invalid(self,ac,ind):
        self.children[ac if ind==0 else slice(None),
                      ac if ind==1 else slice(None)]=False
    def expand(self,x,y):
        new_state = self.state.copy()
        #print("copied",x,y,type(x),type(y),new_state)
        new_state.step(x,y)
        #print("stepped")
        # self.render(new_state)
        s = new_state.get_state()
        valid=1
        if (s[0] == 125 and int(x) != 1):
            valid=0
            self.set_invalid(x,0)
        if (s[232] == 125 and int(y) != 1):
            valid=0
            self.set_invalid(y,1)
        if valid:
            self.add_child((x, y), new_state)
            filled=new_state.check_filled()
            r,w,l=calc_rew(s,(x,y),filled)

            self.children[x, y].is_terminal=w or l
            self.children[x, y].backpropagate(*r)

            return self.children[x, y]
    def best_child(self, c_param=0.2):
        if self.visits==0:
            raise ValueError
        sum_r1 = np.zeros(10)
        sum_r2 = np.zeros(10)
        sum_p1 = np.zeros(10)
        sum_p2 = np.zeros(10)

        invalid_p1 = np.zeros(10)
        invalid_p2 = np.zeros(10)

        for i in range(10):
            for j in range(10):
                c = self.children[i, j]
                if c:
                    sum_r1[i] += c.reward1
                    sum_r2[j] += c.reward2
                    sum_p1[i] += c.visits
                    sum_p2[j] += c.visits
                else:
                    invalid_p1[i] += 1
                    invalid_p2[j] += 1

        # Filter out invalid actions
        valid_indices_p1 = np.where(invalid_p1 != 10)[0]
        valid_indices_p2 = np.where(invalid_p2 != 10)[0]

        sum_p1[invalid_p1 == 10] = np.nan
        sum_p2[invalid_p2 == 10] = np.nan

        sum_p1[valid_indices_p1] = np.maximum(sum_p1[valid_indices_p1], 1)
        sum_p2[valid_indices_p2] = np.maximum(sum_p2[valid_indices_p2], 1)

        ucb1 = (sum_r1 / sum_p1) + c_param * np.sqrt(( np.log(self.visits) / sum_p1))
        ucb2 = (sum_r2 / sum_p2) + c_param * np.sqrt(( np.log(self.visits) / sum_p2))

        # Find the index of the maximum UCB value among valid indices
        try:
            best_move_p1 = valid_indices_p1[np.argmax(ucb1[valid_indices_p1])]
            best_move_p2 = valid_indices_p2[np.argmax(ucb2[valid_indices_p2])]
        except:
            print(self.children)
            breakpoint()
        #print(sum_r1,sum_r2,best_move_p1,best_move_p2)
        return (best_move_p1, best_move_p2)
    def sumr(self):

        r1 = np.array([np.nanmean([c.reward1/c.visits for c in x if c])
             if x.any() else np.nan for x in self.children])

        r2 = np.array([np.nanmean([c.reward2/c.visits for c in x if c])
             if x.any() else np.nan for x in self.children.T])

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
    def __init__(self, i,render,data,results=None,lock=None):
        self.rendering = render and i==0
        self.data = np.frombuffer(data)[i*500:(i+1)*500]
        self.i=i
        self.lock=lock
        self.results=results
    def search(self, root, results=None,remote=None, budget=10000):

        if results:
            self.results=results
        self.remote = remote
        if type(root) == Node:
            self.root = root
            first=False
        else:
            self.root = Node(root)
            first=True

        self.r =tetris.Renderer(1,10) if self.rendering else None
        for it in range(budget):
            leaf,acs = self.traverse(self.root)
            if not leaf.is_terminal:
                simulation_reward,depth = self.rollout(leaf.state, acs,leaf.depth)
                leaf.backpropagate(*simulation_reward)
            else:
                leaf.parent().backpropagate(leaf.reward1,leaf.reward2)
            """
            if depth-leaf.depth<5 and args.debug and it>20:
                print(depth-leaf.depth)
                for _ in range(leaf.depth-depth+1):
                    if leaf.parent is None:
                        break
                    leaf=leaf.parent()
                print(np.array([
                [
                 np.nansum(leaf.children[i,j].reward1)
                if leaf.children[i,j]
                else np.nan
                for j in range(10)
                ]
                for i in range(10)
                ]))
                print(np.array([
                [
                 np.nansum(leaf.children[j,i].reward2)
                if leaf.children[j,i]
                else np.nan
                for j in range(10)
                ]
                for i in range(10)
                ]))
                print(leaf.best_child())
            """
        if self.results is not None:
            result =([sum([c.visits/self.root.visits for c in x if c])
                    if x.any() else 0 for x in self.root.children],
                     [sum([c.visits/self.root.visits for c in x if c])
                    if x.any() else 0 for x in self.root.children.T])
            with self.lock:
                self.results.append((self.root.state.get_state(),result))
        #print(f"reward p1={self.root.reward1/self.root.visits:.3f} p2={self.root.reward2/self.root.visits:.3f}")
        if (first and self.remote):
            self.finalize(budget//10)
        else:
            del self.root

            if self.r:
                self.r.close()
                self.r=None

    def traverse(self, node,d=0):
        init_count=node.depth
        while np.any(node.children) and np.all(np.any(node.children!=None,axis=0)) and np.all(np.any(node.children!=None,axis=1)):
            #print("entered while")
            try:
                acs=node.best_child()
            except:
                print(node.children)
                raise
            if acs:
                if node.children[acs]==None:
                    new_node= node.expand(*acs)
                    if new_node:
                        return new_node,acs
                elif node.children[acs]==False:
                    print(acs)
                    print(node.children)
                    raise ValueError("invalid action on traverse")
                else:
                    node = node.children[acs]
                # self.render(node.state)
            else:
                print(acs)
                print(node.children)
                raise RecursionError("no action found")
        if init_count==node.depth and d:
            print(node.children)
            print(acs)
            raise RecursionError
        if node.is_terminal or np.all(node.children == False):
            node.is_terminal=True
            return node,acs
        else:
            arr=np.array(node.children!=False,dtype=bool)
            while True:
                if self.remote:
                    state=node.state
                    self.data[:]=state.get_state()
                    self.remote.send((self.i,arr))
                    x,y = self.remote.recv()
                else:
                    true_indices = np.argwhere(arr)
                    x,y = true_indices[np.random.randint(len(true_indices))]
                new_node=node.expand(x,y)
                if new_node:
                    #print("return")
                    return new_node,(x,y)

                if np.all(arr) and np.any(node.children) and (np.all(np.any(node.children!=None,axis=0)) and np.all(np.any(node.children!=None,axis=1))):
                    return self.traverse(node,d=1)

    def rollout(self, state,actions, depth):
        is_terminal = False
        reward = np.array([0., 0.])
        init_depth=depth
        while not is_terminal:
            depth += 1

            x = state.get_state()
            if x[8]<0 or x[8]>6:
                print("depth=",depth-init_depth,x)
                print(lastx)
            lastx=x
            filled=state.check_filled()
            r,w,l=calc_rew(x,actions,filled)

            reward+=r
            is_terminal=w or l
            if not is_terminal:
                if self.remote:
                    self.data[:]=x


                    self.remote.send((self.i,np.tile(True,(10,10))))
                    actions = self.remote.recv()
                else:
                    actions=np.random.randint(10,size=2)
                state.step(*actions)

                self.render(state)

        return reward,depth

    def render(self,state):
        if self.r:
            try:
                r = self.r.render(state)
            except:
                print("render error")
                r=1
            if r:
                #self.rendering = False
                self.r.close()
                self.r=None
    def finalize(self,budget=None):
        if not budget:
            del self.root
            return
        keep=[]
        nodes = self.root.collect_nodes()
        nodes.remove(self.root)
        all_rew = np.array([node.sumr() if np.any(node.children) else np.tile(np.nan,(2,10)) for node in nodes])
        not_all_nan=np.where(np.count_nonzero(~np.isnan(all_rew),(1,2))>2)
        if not np.any(not_all_nan):
            #print("nothing to keep")
            del self.root
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
            del self.root
            return
            #raise Exception("filter too high")
        self.remote.send(([k.state.get_state() for k in keep],None))
        p1,p2=self.remote.recv()
        interest=np.nanmax(np.array((np.nansum(adv[:,0]*p1,-1),np.nansum(adv[:,1]*p2,-1))),0)
        #print(f"{interest=}")
        selection=np.array(keep)[interest>3.5]
        #print(f"{len(selection)} interesting states")
        del keep,self.root
        for sel in selection:
            sel.parent=None
            self.search(sel,results=self.results,remote=self.remote,budget=budget)

def calc_rew(x,actions,filled):

    lose = (x[0] == 127)
    win = (x[0] == 126)
    is_terminal = win or lose

    reward=np.array(filled)
    reward=np.minimum(reward,0.7)*0.6
    for i, (r,g) in enumerate(((x[0], x[5]),(x[232],x[237]))):
        if r < 125:
            lines = (r % 10)//2
            atk = r//10
            # hard drop without breaking b2b or spin
            reward[i] += (0.01 if actions[i] ==1
                          else (0.2 if x[i*232+8] == 5 else 0.1))*(r % 2)
            # t spin, regular, tetris
            reward[i] += ((1 if atk >= lines*2+2 and lines <4
                           else 0.3 if lines < 4 else 0.5)*(lines+1.5*atk)**1.2)/5
            # garbage
            reward[i]+=g*0.5
    if is_terminal:
        reward += np.array([10., lose_rew] if win else [lose_rew, 10.]
                           if (x[232] == 126) else [lose_rew, lose_rew])/10
    return reward,win,lose
