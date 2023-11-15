import numpy as np

# ucb factor
C = np.sqrt(2)


class MCTS:
    def __init__(self, env, network, config):
        """Initializes the Monte Carlo Tree Search (MCTS) algorithm.

        :param env: Environment for the game.
        :type env: object
        :param network: Neural network model used to predict action
            probabilities and values.
        :type network: object
        :param config: dicotnary containign config parameters
        :type num_traverses: dictonary
        """
        self.env = env
        self.network = network
        self.num_traverses = config["number search traverses"]
        # each node is identified by its board state string
        self.nodes = {
            self.env.string_board_hist[-1]: {"n":0, "Q":0, "children":{}}
        }

    def get_action_probs(self, pit_mode=False) -> np.ndarray:
        """Gets the probabilities for each action by performing MCTS.

        :param pit_mode: Whether or not to operate in pit mode
        :type pit_mode: bool
        :return: Probabilities for each action
        :rtype: np.ndarray
        """
        # If in pit mode, reset the MCTS tree TODO??
        if pit_mode:
            self.reset()
        
        # Perform MCTS traversals
        for _ in range(self.num_traverses):
            # Create a copy of the environment for search
            search_env = self.env.create_copy()
            self.search(search_env)
        
        # Get the current node in the MCTS tree
        current_node = self.nodes[self.env.string_board_hist[-1]]
        # Calculate child node visit counts
        child_n = []
        for action in range(self.env.action_space_size):
            if action in current_node["children"].keys():
                prob_and_id = current_node["children"][action]
                if len(prob_and_id) > 1:
                    child_n.append(self.nodes[prob_and_id[1]]["n"])
                else:
                    child_n.append(0)
            else:
                child_n.append(0)
        
        # Calculate probabilities based on child node visit counts
        sum_child_n = sum(child_n)
        if sum_child_n == 0:
            # If all probabilities are the same, return a uniform distribution
            child_n = [1/len(child_n) for _ in range(len(child_n))]
            print("ALL probabilites are the same")
            return child_n
        # Calculate move probabilities
        move_probs = np.array(child_n) / sum_child_n
        return move_probs
    

    def search(self, env) -> None:
        """Performs a Monte Carlo Tree Search (MCTS) starting from the
        current state of the environment. The algorithm descends to
        a leaf node, allowing the neural network to estimate the action
        probabilities and leaf value of that node.

        :param env: Environment for the game.
        :type env: object
        """
        node_id = env.string_board_hist[-1]
        # Traverse the tree until a leaf node is reached
        while True:
            node, node_id = self.get_node(env)
            if self.is_leaf_node(node):
                #print("Break")
                break
            mask = env.env.get_legal_actions()
            action = self.get_best_action(node, mask)
            # Check if the chosen action is part of the legal actions
            if not mask[action]:
                print("ACTION NOT PART OF MASK!!!!!!!!!!!!!")
            # Execute the chosen action in the environment
            env.execute_step(action)
    
        # Estimate action probabilities and leaf value using the neural network
        action_probs, leaf_value = self.network(env.get_neutral_board())
        action_probs, leaf_value = np.array(action_probs[0]), leaf_value[0]
        # Apply legal action mask to action probabilities
        valid_moves = np.array(env.legal_actions)
        action_probs *= valid_moves
        # Add child nodes to the tree if the environment is not terminal
        if not env.is_terminal():
            self.add_children(node_id, action_probs, valid_moves)
        # Update Q-values based on the leaf value and move history
        move_hist = env.string_board_hist
        self.update_Qs(-leaf_value, move_hist)


    def get_node(self, env):
        """Gets or creates a node for the current state of the environment.

        :param env: Environment for the game.
        :type env: object
        :return: Node and its identifier.
        :rtype: dict, string
        """
        state_string = env.string_board_hist[-1]
        prev_state_id = env.get_previous_board_string()
        # Create a new node if the current state is not in the tree
        if state_string not in self.nodes.keys():
            self.nodes[state_string] = {"n":0, "Q":0, "children":{}}
        # Update the children of the parent node
        if prev_state_id in self.nodes:
            prev_action = env.action_acc[-1]
            p_and_id = self.nodes[prev_state_id]["children"][prev_action]
            # Check if parent already holds its childs id
            if len(p_and_id) == 1:
                # Add childs id to parent
                p_and_id.append(state_string)
                self.nodes[prev_state_id]["children"][prev_action] = p_and_id
        return self.nodes[state_string], state_string
    

    def is_leaf_node(self, node) -> bool:
        """Checks if the current node is a leaf node (has no children).

        :param node: Node to be checked
        :type node: dict
        :returns: True if the node is a leaf node, False otherwise
        :rtype: bool
        """
        return node["children"] == {}
    

    def get_best_action(self, node, mask: np.ndarray) -> int:
        """Gets the best action based on the UCB values, considering
        legal actions.

        :param node: Node information containing visit counts,
            Q-values, and child nodes
        :type node: dict
        :param mask: Mask that identifies the legal actions
        :type mask: np.ndarray
        :return: Index of the best action
        :rtype: int
        """
        ucbs = np.zeros_like(mask)
        # Number of times the node was visited
        node_n = node["n"]
        for action, prob_child_id in node["children"].items():
            if len(prob_child_id) == 1:
                prob = prob_child_id[0]
                ucbs[action] = prob * np.sqrt(node_n + 1e-8)
            else:
                prob = prob_child_id[0]
                child = self.nodes[prob_child_id[1]]
                child_n = child["n"]
                child_Q = child["Q"]
                ucbs[action] = child_Q + prob * np.sqrt(node_n) / (1 + child_n)
        # Set UCB values for illegal actions to negative infinity    
        ucbs[mask==0] = -np.inf
        # Return the index of the action with the highest UCB value
        return np.argmax(ucbs)
    

    def add_children(self, node_id, probs: np.ndarray, mask) -> None:
        """Adds action to a node and the associated probablity.

        :param node_id: Identifier of the current node
        :type node_id: str
        :param probs: Probabilities associated with each action
        :type probs: np.ndarray
        :param mask: Mask that identifies the legal actions
        :type mask: np.ndarray
        """
        for action, p in enumerate(probs):
            # Add child probablity only if the action is legal
            if mask[action]:
                self.nodes[node_id]["children"][action] = [p]
    

    def update_Qs(self, leaf_value: int, move_hist: list):
        """Updates the Q-values for nodes in the move history.

        :param leaf_value: Value of the leaf node to update Q-values
        :type leaf_value: int
        :param move_hist: List of board states in the move history
        :type move_hist: list
        """
        # Iterate through the move history in reverse order
        for node_id in move_hist[::-1]:
            # Check if the node is in the tree
            if node_id in self.nodes.keys():
                node = self.nodes[node_id]
                # Increment the visit count (n) and update the Q-value
                self.nodes[node_id]["n"] += 1
                self.nodes[node_id]["Q"] += (leaf_value - node["Q"]) / node["n"]
            # Reverse the sign of the leaf value for alternating players
            leaf_value *= -1


    def reset(self) -> None:
        """Resets the root node of the search tree."""
        self.nodes = {
            self.env.string_board_hist[-1]: {"n":0, "Q":0, "children":{}}
        }