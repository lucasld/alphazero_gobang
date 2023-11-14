import numpy as np

from typing import Union
import time
import copy

C = np.sqrt(4)  # ucb factor  #TODO put this somewhere else


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
        # each node is identified by its board state string
        self.nodes = {self.env.string_board_hist[-1]:
                          {"n":0, "Q":0, "children":{}}
                     }

    def get_action_probs(self, pit_mode=False) -> np.ndarray:
        """Gets the probabilities for each action by performing MCTS.

        :param pit_mode: Whether or not to operate in pit mode.
        :type pit_mode: bool
        :returns: Probabilities for each action.
        :rtype: np.ndarray
        """
        #print("####"*20, "before", len(self.nodes.keys()))
        if pit_mode:
            self.reset()
        #print()
        #print("aaa")
        search_env = self.env.create_copy()
        for _ in range(self.num_traverses):
            #search_env.copy_values_over(self.env)
            search_env = self.env.create_copy()
            self.search(search_env)
        #print("bbb")
        #print()
        
        current_node = self.nodes[self.env.string_board_hist[-1]]
        #print("CURRENT NODE", current_node)
        child_n = []
        for action in range(self.env.action_space_size):
            if action in current_node["children"].keys():
                #print("Action", action)
                prob_and_id = current_node["children"][action]
                if len(prob_and_id) > 1:
                    child_n.append(self.nodes[prob_and_id[1]]["n"])
                else:
                    child_n.append(0)
                #print(child_n)
            else:
                child_n.append(0)
        
        sum_child_n = sum(child_n)
        #print("####"*20, "after", len(self.nodes.keys()))
        if sum_child_n == 0:
            child_n = [1/len(child_n) for _ in range(len(child_n))]
            #print("ALL probabilites are the same")
            return child_n
        move_probs = np.array(child_n) / sum_child_n  #TODO: softmax?
        return move_probs
    

    def search(self, env) -> None:
        """Performs a search in the tree starting from the current state
        of the environment. The algorithm goes down to a leaf node and lets the neural network
        estimate the action probs and leaf value of that node.
        
        TODO: add type here

        :param env: Environment for the game.
        :type env: object
        """
        #self.visualize_board(env)
        #print("SEARCH")
        node_id = env.string_board_hist[-1]
        original_node = copy.deepcopy(node_id)
        while True:
            node, node_id = self.get_node(env)
            if self.is_leaf_node(node):
                #print("Break")
                break
            mask = env.env.get_legal_actions()
            action = self.get_best_action(node, mask)
            if not mask[action]:
                pass
                #print("ACTION NOT PART OF MASK!!!!!!!!!!!!!!!!!!!!!!!!!")
            #print("taking action", action)
            env.execute_step(action)
        
        action_probs, leaf_value = self.network(env.board.reshape(1, 5, 5, 1))
        action_probs, leaf_value = np.array(action_probs[0]), leaf_value[0]
        valid_moves = np.array(env.legal_actions)
        action_probs *= valid_moves
        if not env.is_terminal():
            self.add_children(node_id, action_probs, valid_moves)

        move_hist = env.string_board_hist
        #print("ORIGINAL NODE IN MOVE HIST" if original_node in move_hist else "...................")
        self.update_Qs(-leaf_value, move_hist)

    def get_node(self, env):
        state_string = env.string_board_hist[-1]
        if state_string not in self.nodes.keys():
            self.nodes[state_string] = {"n":0, "Q":0, "children":{}}
            prev_state_id = env.get_previous_board_string()
            if prev_state_id in self.nodes:
                prev_action = env.action_acc[-1]
                p = self.nodes[prev_state_id]["children"][prev_action]
                if len(p) == 1:
                    p.append(state_string)
                    self.nodes[prev_state_id]["children"][prev_action] = p
        return self.nodes[state_string], state_string
    

    def is_leaf_node(self, node) -> bool:
        """Checks if the current node is a leaf node (has no children).

        :returns: True if the node is a leaf node, False otherwise.
        :rtype: bool
        """
        return node["children"] == {}
    

    def get_best_action(self, node, mask: np.ndarray) -> int:
        """Gets the best action based on the UCB values, considering
        legal actions.

        :param mask: Mask that identifies the legal actions.
        :type mask: np.ndarray
        :returns: Index of the best action.
        :rtype: int
        """
        ucbs = np.zeros_like(mask)
        node_n = node["n"]  # number of times the node was visited
        #print(node)
        for _action, prob_child_id in node["children"].items():
            #print("AA", _action)
            if len(prob_child_id):
                prob = prob_child_id[0]
                child_n = 0
                child_Q = 0
            else:
                child = self.nodes[prob_child_id[1]]
                child_n = child["n"]
                child_Q = child["Q"]
            ucbs[_action] = self.ucb(child_n, child_Q, prob, node_n)        
        ucbs[mask==0] = -np.inf
        #print(mask, np.round(ucbs))
        #input()
        return np.argmax(ucbs)
    
    

    def ucb(self, node_n, node_Q, p, parent_node_n) -> float:
        """Calculates the Upper Confidence Bound (UCB) for a node.

        :param node: node for which ucb should be calculated
        :type node: dict
        :param p: probablity that this node was chosen from the previous node
        :type p: float
        :param parent_node_n: number of times parent node was called
        :type parent_node_n: int
        :returns: UCB value for the node.
        :rtype: float
        """
        u = C * p * np.sqrt(parent_node_n) / (1 + node_n)
        return node_Q + u
    

    '''def add_children(self, node_id, probs: np.ndarray, mask) -> None:
        """Adds child nodes to the current node for given action probabilities.

        :param probs: Probabilities associated with each action.
        :type probs: np.ndarray
        """
        for action, p in enumerate(probs):
            if mask[action] and action not in self.nodes[node_id]["children"].keys():
                # child prototype which will later be added to self.nodes
                child_prototype = {"Q":0, "n":0, "children":{}}
                self.nodes[node_id]["children"][action] = [p, child_prototype]'''
    def add_children(self, node_id, probs: np.ndarray, mask) -> None:
        """Adds child nodes to the current node for given action probabilities.

        :param probs: Probabilities associated with each action.
        :type probs: np.ndarray
        """
        for action, p in enumerate(probs):
            if mask[action]:
                self.nodes[node_id]["children"][action] = [p]
    
    def update_Qs(self, leaf_value: int, move_hist: list):
        """Updates the Q-values.

        :param leaf_value: Value of the leaf node to update Q-values.
        :type leaf_value: integer
        """
        #print("UPDATING")
        #print(move_hist[0] in self.nodes.keys())
        for node_id in move_hist[::-1]:
            if node_id  in self.nodes.keys():
                node = self.nodes[node_id]
                self.nodes[node_id]["n"] += 1
                self.nodes[node_id]["Q"] += (leaf_value - node["Q"]) / node["n"]
            leaf_value *= -1

    

    def reset(self) -> None:
        #print("MCTS reset!!!!!!!!!!!")
        """Resets the root node of the search tree."""
        self.nodes = {self.env.string_board_hist[-1]:
                          {"n":0, "Q":0, "children":{}}
                     }
    

    def visualize_board(self, env):
        print("")
        for row in range(env.height):
            row_list = []
            for e in env.board[row]:
                if e == -1:
                    row_list.append("X")
                elif e == 1:
                    row_list.append("O")
                else:
                    row_list.append(" ")
            print(" | ".join(row_list))
            if row < env.height - 1:
                print("-"*int(len(row_list)*3.7))
        print("")