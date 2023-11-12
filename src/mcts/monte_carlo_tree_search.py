import numpy as np

from typing import Union
import time


C = np.sqrt(2)  # ucb factor  #TODO put this somewhere else


class MCTS:
    def __init__(self, env, network, config):
        """Initializes the Monte Carlo Tree Search (MCTS) algorithm.
        TODO: add types here

        :param env: Environment for the game.
        :type env: object
        :param network: Neural network model used to predict action
            probabilities and values.
        :type network: object
        :param num_traverses: Number of traversals in the search tree.
        :type num_traverses: int
        """
        self.env = env
        self.network = network
        self.num_traverses = config["number search traverses"]
        
        self.root_node = Node(parent_node=False, p=1.0)

    def get_action_probs(self, pit_mode=False) -> np.ndarray:
        """Gets the probabilities for each action by performing MCTS.

        :param pit_mode: Whether or not to operate in pit mode.
        :type pit_mode: bool
        :returns: Probabilities for each action.
        :rtype: np.ndarray
        """
        t1 = time.time()
        if pit_mode: self.reset()
        t2 = time.time()
        for _ in range(self.num_traverses):
            env_temp = self.env.create_copy()
            #copyt = time.time()
            #print("copy()", copyt - t2)
            #if pit_mode: env_temp.action_acc = []
            self.search(env_temp)
            #t2 = time.time()
            #print("search()", t2 - copyt)

        #state_string = self.env.get_state_string()
        child_n = np.array([child.n for child in self.get_node(self.env).children.values()])
        move_probs = np.array(child_n) / sum(child_n)  #TODO: softmax?
        print("get_action_probs():", time.time() - t1)
        return move_probs
    

    def search(self, env) -> None:
        """Performs a search in the tree starting from the current state
        of the environment. The algorithm goes down to a leaf node and lets the neural network
        estimate the action probs and leaf value of that node.
        
        TODO: add type here

        :param env: Environment for the game.
        :type env: object
        """
        #s = env.get_state_string()
        node = self.get_node(env)#TODO:self.root_node
        while True:
            if node.is_leaf_node():
                break
            # choose action with highest ucb value
            mask = env.legal_actions
            action = node.get_best_action(mask)
            node = node.children[action]
            env.execute_step(action)
        action_probs, leaf_value = self.network(env.board.reshape(1, 5, 5, 1))
        action_probs, leaf_value = np.array(action_probs[0]), leaf_value[0]
        valid_moves = np.array(env.legal_actions)
        action_probs *= valid_moves
        if not env.is_terminal():
            node.add_children(action_probs)
        # TODO check if leaf value has to be changed
        node.update_Qs(-leaf_value)
    

    def reset(self) -> None:
        """Resets the root node of the search tree."""
        self.root_node = Node(parent_node=False, p=1.0)

    
    def get_node(self, env):
        """Retrieves the node corresponding to the current state
        of the environment.
        TODO: add types here

        :param env: Environment for the game.
        :type env: object
        :returns: Corresponding node in the search tree.
        :rtype: Node object
        """
        n = self.root_node
        for a in env.action_acc:
            if a not in n.children.keys():
                n.children[a] = Node(n, 1.0)
            n = n.children[a]
        return n


class Node:
    def __init__(self, parent_node: Union['Node', bool], p: float):
        """Initializes a node in the search tree.

        :param parent_node: Parent of the current node (False if it is
            the root node).
        :type parent_node: Node or bool
        :param p: Probability associated with the action that led to this node.
        :type p: float
        """
        self.Q = 0  # running average of values for all visits to this node
        self.u = 0  # exploration term of the ucb-formula
        self.n = 0  # number of visits to this node
        # if this is the rood node parent_node is set to False
        self.parent_node = parent_node
        # as key the action and as value the child-node
        self.children = {}
        # action probablilites
        self.p = p
    

    def add_children(self, probs: np.ndarray) -> None:
        """Adds child nodes to the current node for given action probabilities.

        :param probs: Probabilities associated with each action.
        :type probs: np.ndarray
        """
        for action, p in enumerate(probs):
            if action not in self.children.keys():
                self.children[action] = Node(parent_node=self, p=p)
    

    def is_leaf_node(self) -> bool:
        """Checks if the current node is a leaf node (has no children).

        :returns: True if the node is a leaf node, False otherwise.
        :rtype: bool
        """
        return self.children == {}
    

    def ucb(self) -> float:
        """Calculates the Upper Confidence Bound (UCB) for the node.

        :returns: UCB value for the node.
        :rtype: float
        """
        self.u = C * self.p * np.sqrt(self.parent_node.n) / (1 + self.n)
        return self.Q + self.u
    

    def update_Qs(self, leaf_value: int):
        """Updates the Q-value of the current node and recursively updates
        the parent nodes.

        :param leaf_value: Value of the leaf node to update Q-values.
        :type leaf_value: integer
        """
        self.n += 1
        self.Q += (leaf_value - self.Q) / self.n
        if self.parent_node:
            self.parent_node.update_Qs(-leaf_value)
    

    def get_best_action(self, mask: np.ndarray) -> int:
        """Gets the best action based on the UCB values, considering
        legal actions.

        :param mask: Mask that identifies the legal actions.
        :type mask: np.ndarray
        :returns: Index of the best action.
        :rtype: int
        """
        ucbs = np.array([float(child.ucb()) for child in self.children.values()])
        ucbs[mask==0] = -1
        return np.argmax(ucbs)