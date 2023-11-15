import numpy as np
from src.utils import tools


class Environment:
    def __init__(self, config):
        """Initializes the Gobang Environment Manager.
        
        :param config: configuration dictionary containing game parameters
        :type config: dict
        """
        # Extracting game parameters from the configuration dictionary
        self.config = config
        self.height = config["height"]
        self.width = config["width"]
        self.required_pieces = config["number win pieces"]
        self.pit_number = config["number pits"]
        # Initializing the Gobang environment object
        self.env = Gobang_Env(width=self.width, height=self.height,
                              required_win_pieces=self.required_pieces)
        # List that accumulates all actions taken in the environment
        self.action_acc = []
        # History of string representations of the game board
        self.string_board_hist = [self.get_board_string()]
        # Resetting the environment to its initial state
        self.reset_env()
        # Calculating the size of the action space
        self.action_space_size = self.width * self.height
        # Path to the experience replay directory in the configuration
        self.hist_dir_path = self.config['experience path']
    
    
    def execute_step(self, action):
        """Executes a step in the Gobang environment.

        :param action: the action to be taken in the environment
        :type action: int
        :return: True if the step was successfully executed, False otherwise
        :rtype: bool
        """
        # Remember the board state string before taking the step
        # Take a step in the environment
        successful = self.env.step(action)
        # Update attributes with the current state of the environment
        self.board, self.legal_actions, self.terminal, self.reward = self.env.last()
        # Add the action to the action accumulator
        self.action_acc.append(action)
        # Update the history of string representations of the game board
        self.string_board_hist.append(self.get_board_string())
        return successful

    
    def is_terminal(self):
        """Returns true if game is either truncated or terminated.
        
        :return: if game is terminated true, else false
        :rtype: boolean
        """
        return self.env.is_terminal()


    def pit(self, new_agent, old_agent, win_treshold):
        """Pits two agents against each other for a specified number of games.

        :param new_agent: The agent representing the new network.
        :type new_agent: function get_action_prob()
        :param old_agent: The agent representing the old network.
        :type old_agent: function get_action_prob()
        :param win_threshold: The threshold for victory, value between 0 and 1.
        :type win_threshold: float
        :return: True if the new agent wins, False otherwise.
        :rtype: bool
        """
        print("Start pitting...")
        # Set up the agents and counters for wins and draws
        self.agents = (new_agent, old_agent)
        self.new_agent_wins, self.old_agent_wins, self.draws = 0, 0, 0
        # Iterate through the specified number of games
        for pit_i in range(self.pit_number):
            print("Pit number:", pit_i + 1, "/", self.pit_number)
            # Determine which player starts the game
            player_i = 0 if pit_i%2 else 1
            print(f"{'old' if player_i else 'new'} player starting the game.. {player_i}")
            # Play a game between the two agents
            experience_replay = self.execute_pit(player_i)
            # Add new examples to the experience replay if a path is specified
            if self.hist_dir_path:
                tools.add_new_examples(self.hist_dir_path, experience_replay)
            # Create a gif to visualize the game
            if pit_i < 10:
                tools.create_gif(f"pit_n{pit_i}", self.action_acc, self.config)
            # Update win counters based on the game outcome
            if player_i == 0:
                if self.reward == 0:
                    self.draws += 1
                elif self.reward == 1:
                    self.new_agent_wins += 1
                    print("black wins!")
                elif self.reward == -1:
                    print("white wins")
                    self.old_agent_wins += 1
            else:
                if self.reward == 0:
                    self.draws += 1
                elif self.reward == 1:
                    self.old_agent_wins += 1
                    print("black wins")
                elif self.reward == -1:
                    self.new_agent_wins += 1
                    print("white wins")
            # Check if the new agent has already lost
            rounds_left = self.pit_number - pit_i - 1
            current_new_win_threshold = (self.old_agent_wins / (1 - win_treshold)) * win_treshold
            print(self.new_agent_wins, self.old_agent_wins, self.draws)
            if self.new_agent_wins + rounds_left < current_new_win_threshold:
                return False
            # Check if the old agent has already lost
            current_old_win_threshold = (self.new_agent_wins / win_treshold) * (1 - win_treshold)
            if self.old_agent_wins + rounds_left <= current_old_win_threshold:
                return True
    

    def execute_pit(self, player_i):
        """Executes one episode between two different agents/players.
        
        :param player_i: represents the player that starts (black) with the first
            move, either 0 or 1
        :type player: integer
        :returns: experience replay
        :rtype: list
        """
        experience_replay = []
        self.reset_env()
        self.agents[player_i].reset()
        move_count = 0
        while not self.is_terminal():
            # Compute action probablities by performing tree search
            pi = self.agents[player_i].get_action_probs()
            # Add a "neutralized" board to the experience replay (a board which
            # is player invariant)
            experience_replay.append((self.get_neutral_board(), pi, None))
            # Mask pi so that only legal moves are > 0
            mask = self.legal_actions
            masked_pi = pi
            masked_pi[mask==0] = 0
            # Choose the action with the highest probability
            action = np.argmax(masked_pi)
            # Choose random action for the first two moves so that different
            # games happen
            if move_count < 2:
                action = np.random.choice(len(self.legal_actions), p=mask/(sum(mask)))
            # Take action
            self.execute_step(action)
            player_i = int(not player_i)
            move_count += 1
        # Assign rewards to each player's move
        for player_i, (observation, pi, _) in list(enumerate(experience_replay))[::-1]:
            r = self.env.reward * (1 if player_i%2 else -1)
            experience_replay[player_i] = (observation, pi, r)
        return experience_replay
    

    def reset_env(self):
        """Resets environment"""
        self.env.board = np.zeros((self.height, self.width))
        self.env.player = 0
        self.env.reward = 0
        self.env.legal_actions = self.env.get_legal_actions()
        self.env.terminal = False
        self.board, self.legal_actions, self.terminal, self.reward = self.env.last()
        self.action_acc = []
        self.string_board_hist = [self.get_board_string()]


    def create_copy(self):
        """Function that creates a copy of the environment itself."""
        env_copy = Environment(self.config)
        for action in self.action_acc:
            env_copy.execute_step(action)
        return env_copy
    
    
    def get_previous_board_string(self):
        """Retrieves the string representation of the board from the
        previous game state.

        :return: String representation of the board from the previous
            game state, or False if not available.
        :rtype: str or False
        """
        if len(self.string_board_hist) > 1:
            return self.string_board_hist[-2]
        return False


    """def copy_values_over(self, other_env):
        self.config = other_env.config
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.required_pieces = self.config["number win pieces"]
        self.pit_number = self.config["number pits"]
        self.action_acc = copy.deepcopy(other_env.action_acc)
        self.string_board_hist = copy.deepcopy(self.string_board_hist)
        self.env.board = np.copy(other_env.env.board)
        self.env.win_pieces = other_env.env.win_pieces
        self.env.player = copy.deepcopy(other_env.env.player)
        self.env.pieces = other_env.env.pieces
        self.env.reward = copy.deepcopy(other_env.env.reward)
        self.env.legal_actions = other_env.env.get_legal_actions()
        self.env.terminal = copy.deepcopy(other_env.env.terminal)
        self.board, self.legal_actions, self.terminal, self.reward = self.env.last()"""


    def get_neutral_board(self):
        player = self.env.player
        player1 = self.env.board == player
        player2 = self.env.board == int(not player)
        empty_space = self.env.board == 0
        board_new = np.copy(self.env.board)
        board_new[player1] = 0
        board_new[player2] = 1
        board_new[empty_space] = 2
        return board_new


    def get_board_string(self):
        """Returns a string representation of the board. However we dont care
        about the player's color - the player currently playing is denoted with
        0, the enemy pieces are denoted with 1 and empty space with 2.
        """
        board_new = self.get_neutral_board()
        flattened_board = str(board_new.flatten())
        # Create a translation table to remove specified characters
        trans_table = str.maketrans("", "", " .\n[]")
        # Apply the translation to the string
        result = flattened_board.translate(trans_table)
        return result
    

class Gobang_Env:
    def __init__(self, width=19, height=19, required_win_pieces=5):
        """Initializes the Gobang Environment.

        :param width: Width of the game board.
        :type width: int, optional
        :param height: Height of the game board.
        :type height: int, optional
        :param required_win_pieces: Number of pieces in a row/column/diagonal
            required to win.
        :type required_win_pieces: int, optional
        """
        self.width, self.height = width, height
        self.board = np.zeros((height, width))
        self.win_pieces = required_win_pieces
        self.player = 0  # alternates between 0 and 1
        self.pieces = {0: 1, 1: -1}  # key=player, value=piece
        
        self.reward = 0
        self.legal_actions = self.get_legal_actions()
        self.terminal = False


    def step(self, action: int):
        """Take a step in the Gobang environment.

        :param action: The action to be taken, representing a position on the board.
        :type action: int
        :return: True if the step is successful, False otherwise.
        :rtype: bool
        """
        # Action space is a self.width * self.height large 1-dimensional array
        row = action // self.width
        column = action - (row * self.width)
        # Check if action is legal
        if not self.get_legal_actions()[action]:
            print("! Illegal Action:", action, "!")
            print(np.round(self.board))
            print(self.legal_actions.reshape((self.height, self.width)))
            input()
            return False
        # Place piece on the board
        self.board[row, column] = self.pieces[self.player]
        # Remove enemy pieces if caught (2 pieces in between 2 pieces)
        self.remove_enemy(row, column)
        # Update terminal and reward based on the current state
        self.terminal = self.is_terminal()
        if self.terminal:
            if self.is_draw():
                self.reward = 0
            elif self.player == 0:
                self.reward = 1
            elif self.player == 1:
                self.reward = -1
        # Switch player and update legal actions
        self.player = int(not self.player)
        self.legal_actions = self.get_legal_actions()
        return True


    def last(self):
        """Returns the current state information.

        :return: Tuple containing the board, legal actions,
            terminal state and reward.
        :rtype: tuple
        """
        return (self.board, self.legal_actions, self.terminal, self.reward)


    def get_legal_actions(self):
        """Returns an array representing legal moves, where 0 indicates illegal
        moves and 1 indicates legal moves.

        :return: Array representing legal actions.
        :rtype: numpy.ndarray
        """
        action_mask = np.zeros_like(self.board)
        # the first move of the game has to be on the center crosspoint
        if 1 not in self.board:
            action_mask[self.height//2, self.width//2] = 1
            # flatten action mask
            action_mask = action_mask.ravel()
            return action_mask
        # after first move every move is legal where there is no piece
        action_mask[self.board == 0] = 1
        # flatten action mask
        action_mask = action_mask.ravel()
        return action_mask


    def remove_enemy(self, row, column):
        """Checks in every possible direction if the removal condition is met
        and removes enemy pieces accordingly.

        :param row: Row index of the last placed piece.
        :type row: int
        :param column: Column index of the last placed piece.
        :type column: int
        """
        player_color = self.pieces[self.player]
        directions = [(1, 0), (-1, 0),
                      (0, 1),(0, -1),
                      (1, 1), (1, -1),
                      (-1, 1), (-1, -1)]
        for ydir, xdir in directions:
            y, x = row + ydir * 3, column + xdir * 3
            if 0 <= y < self.height and \
               0 <= x < self.width and \
               self.board[y, x] == player_color:
                pattern = []
                ye1, xe1 = row + ydir * 2, column + xdir * 2
                pattern.append(self.board[ye1, xe1])
                ye2, xe2 = row + ydir * 1, column + xdir * 1
                pattern.append(self.board[ye2, xe2])
                enemy_player_color = self.pieces[int(not self.player)]
                # Check if both inner pieces are from the enemy player
                if np.all(np.array(pattern) == enemy_player_color):
                    # Remove enemy pieces
                    self.board[ye1, xe1] = 0
                    self.board[ye2, xe2] = 0
    

    def is_draw(self):
        """Checks if a position is draw."""
        return not np.any(self.board == 0)
    
    
    def is_terminal(self):
        """Checks if a position is terminal"""
        if self.is_draw(): return True

        def check_lines(arr):
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]-self.win_pieces+1):
                    if abs(np.sum(arr[y, x:x+self.win_pieces])) == self.win_pieces:
                        return True
            return False
        
        def check_diagonals(arr):
            for i in range(-arr.shape[0]+1, arr.shape[0]-1):
                diagonal = arr.diagonal(i)
                for x in range(len(diagonal)-self.win_pieces+1):
                    if abs(np.sum(diagonal[x:x+self.win_pieces])) == self.win_pieces:
                        return True
            return False
        
        return any([
            check_lines(self.board),
            check_lines(self.board.T),
            check_diagonals(self.board),
            check_diagonals(np.flipud(self.board))
        ])
    

    def copy_env(self):
        """Creates a deep copy of the current game environment.

        :return: New instance of the Gobang_Env class with copied state
        :rtype: Gobang_Env-object
        """
        new_env = Gobang_Env(self.width, self.height, self.win_pieces)
        new_env.board = np.copy(self.board)
        new_env.player = self.player
        new_env.reward = self.reward
        new_env.legal_actions = new_env.get_legal_actions()
        new_env.terminal = new_env.is_terminal()
        return new_env