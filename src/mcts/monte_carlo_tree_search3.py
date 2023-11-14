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
        self.nodes = {}

    def get_action_probs(self):#, pit_mode=False) -> np.ndarray:
        """Gets the probabilities for each action by performing MCTS.

        :param pit_mode: Whether or not to operate in pit mode.
        :type pit_mode: bool
        :returns: Probabilities for each action.
        :rtype: np.ndarray
        """
        #current_node = self.nodes[self.env.get_board_string()]
        search_env = self.env.create_copy()
        for _ in range(self.num_traverses):
            #search_env.copy_values_over(self.env)
            search_env = self.env.create_copy()
            self.search(search_env)
        node = self.nodes[self.env.get_board_string()]
        print(node)
        counts = [node["Na"][a] if a in node["Na"].keys() else 0 for a in range(self.env.action_space_size)]
        
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
    

    def search(self, env):
        node_id = env.get_board_string()
        if env.env.is_terminal():
            return -env.reward

        if node_id not in self.nodes:
            action_probs, leaf_value = self.network(env.board.reshape(1, 5, 5, 1))
            action_probs, leaf_value = np.array(action_probs[0]), leaf_value[0]
            print("NETWORK", action_probs)
            valid_moves = np.array(env.legal_actions)
            #print(1, action_probs)
            action_probs *= valid_moves
            #print(2, action_probs)
            sum_probs_s = np.sum(action_probs)
            if sum_probs_s > 0:
                action_probs /= sum_probs_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, doing a workaround.")
                action_probs = action_probs + valid_moves
                action_probs /= np.sum(action_probs)
            self.nodes[node_id] = {"action_probs":action_probs, "n":0, "Qa":{}, "Na":{}}
            return -leaf_value
    
        max_u, best_action = -float("inf"), -1
        valid_moves = np.array(env.legal_actions)
        moves = np.where(valid_moves > 0)
        for action in moves:
            action = int(action[0])
            print(action)
            n = self.nodes[node_id]
            if action in self.nodes[node_id]["Qa"].keys():
                u = n["Qa"][action] + C + n["action_probs"][action] * np.sqrt(n["n"]) / (1 + n["Na"][action])
            else:
                u = C * n["action_probs"][action] * np.sqrt(n["n"] + 1e-8)  # Q = 0 ?
            if u > max_u:
                max_u = u
                best_action = action

        a = best_action
        env.execute_step(action)

        v = self.search(env)

        if a in self.nodes[node_id]["Qa"]:
            n = self.nodes[node_id]
            self.nodes[node_id]["Qa"][a] = (n["Na"][a] * n["Qa"][a] + v) / (n["Na"][a] + 1)
            self.nodes[node_id]["Na"][a] += 1
        else:
            self.nodes[node_id]["Qa"][a] = v
            self.nodes[node_id]["Na"][a] = 1

        self.nodes[node_id]["n"] += 1
        return -v
    

    def get_node(self, env):
        state_string = env.get_board_string()
        previous_state_id = env.previous_state_string
        if state_string not in self.nodes.keys():
            if previous_state_id in self.nodes.keys():
                prev_action = env.action_acc[-1]
                # add new state string to children of previous node
                p, child_prototype = copy.deepcopy(self.nodes[previous_state_id]["children"][prev_action])
                self.nodes[previous_state_id]["children"][prev_action] = [p, state_string]
                # add new node
                self.nodes[state_string] = child_prototype
            else:
                print("Previous node was not found")
        if previous_state_id != False:
            for action, (p, child) in self.nodes[previous_state_id]["children"].items():
                if child == state_string:
                    self.nodes[previous_state_id]["children"][action][1] = state_string
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
        for _action, (prob, child_board_string) in node["children"].items():
            #print("AA", _action)
            if type(child_board_string) == dict:
                child = child_board_string
            else:
                child = self.nodes[child_board_string]
            ucbs[_action] = self.ucb(child, prob, node_n)        
        ucbs[mask==0] = -np.inf
        #print(mask, np.round(ucbs))
        #input()
        return np.argmax(ucbs)
    
    

    def ucb(self, node, p, parent_node_n) -> float:
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
        u = C * p * np.sqrt(parent_node_n) / (1 + node["n"])
        return node["Q"] + u
    

    def add_children(self, node_id, probs: np.ndarray, mask) -> None:
        """Adds child nodes to the current node for given action probabilities.

        :param probs: Probabilities associated with each action.
        :type probs: np.ndarray
        """
        for action, p in enumerate(probs):
            if mask[action] and action not in self.nodes[node_id]["children"].keys():
                # child prototype which will later be added to self.nodes
                child_prototype = {"Q":0, "n":0, "children":{}}
                self.nodes[node_id]["children"][action] = [p, child_prototype]

    
    def update_Qs(self, leaf_value: int, move_hist: list):
        """Updates the Q-values.

        :param leaf_value: Value of the leaf node to update Q-values.
        :type leaf_value: integer
        """
        for node_id in move_hist[::-1]:
            node = self.nodes[node_id]
            self.nodes[node_id]["n"] += 1
            self.nodes[node_id]["Q"] += (leaf_value - node["Q"]) / node["n"]
            leaf_value *= -1

    

    def reset(self) -> None:
        print("MCTS reset")
        """Resets the root node of the search tree."""
        self.nodes = {}
    

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