import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import io
import copy
from src.utils import tools


class Environment:
    def __init__(self, config):
        self.config = config
        self.height = config["height"]
        self.width = config["width"]
        self.required_pieces = config["number win pieces"]
        self.pit_number = config["number pits"]
        self.env = Gobang_Env(width=self.width, height=self.height,
                              required_win_pieces=self.required_pieces)
        # list that accumulates all actions taken to create on trajectory
        self.action_acc = []
        self.string_board_hist = [self.get_board_string()]
        self.reset_env()
        self.action_space_size = self.width * self.height
        self.hist_dir_path = self.config['experience path']
    
    
    def execute_step(self, action):
        # remeber board state string before taking step
        # take a step in the environment
        successful = self.env.step(action)
        self.board, self.legal_actions, self.terminal, self.reward = self.env.last()
        # add action to action accumulator
        self.action_acc.append(action)
        self.string_board_hist.append(self.get_board_string())
        return successful

    
    def is_terminal(self):
        """Returns true if game is either truncated or terminated.
        
        :return: if game is terminated true, else false
        :rtype: boolean
        """
        return self.env.is_terminal()


    def pit(self, new_agent, old_agent, win_treshold):
        """Pits to agents against each other self.pit_number times.
        
        :param agent1: agent 1, MCTS-oject
        :type agent1: MCTS-object
        :param agent2: agent 2, MCTS-oject
        :type agent2: MCTS-object"""
        print("Start pitting...")
        self.agents = (new_agent, old_agent)
        self.new_agent_wins, self.old_agent_wins, self.draws = 0, 0, 0
        # starting games with new agent
        for pit_i in range(self.pit_number):
            print("pit_i:", pit_i, "/", self.pit_number)
            # play a game between the two agents
            player = 0 if pit_i%2 else 1
            print(f"{'old' if player else 'new'} player starting the game..")
            experience_replay = self.execute_pit(player)
            if self.hist_dir_path:
                tools.add_new_examples(self.hist_dir_path, experience_replay)
            self.create_gif(f"pit_n{pit_i}")
            if player == 0:
                # adding win to respective player
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
            # check if it is still possible for the new network to win
            # number of rounds left in the game
            rounds_left = self.pit_number//2 - pit_i - 1
            # number of wins the new network has to achieve at least to win
            # check if it is still possible for the new network to win
            # number of rounds left in the game
            rounds_left = self.pit_number - pit_i - 1
            # number of wins the new network has to achieve at least to win
            current_new_win_threshold = (self.old_agent_wins / (1 - win_treshold)) * win_treshold
            print(self.new_agent_wins, self.old_agent_wins, self.draws)
            if self.new_agent_wins + rounds_left < current_new_win_threshold:
                return False
            # check if it is still possible for the old network to win
            current_old_win_threshold = (self.new_agent_wins / win_treshold) * (1 - win_treshold)
            if self.old_agent_wins + rounds_left < current_old_win_threshold:
                return True
        # check who won
        if self.new_agent_wins + self.old_agent_wins == 0 or self.new_agent_wins / (self.new_agent_wins + self.old_agent_wins) < self.win_prob_threshold:
            print(self.new_agent_wins, self.old_agent_wins, self.draws)
            return False
        else:
            print(self.new_agent_wins, self.old_agent_wins, self.draws)
            return True
    

    def execute_pit(self, player):
        """Executing one episode between two different agents/players.
        
        :param player: represents the player that starts (black) with the first
            move, either 0 or 1
        :type player: integer
        :returns: player that won, experience replay
        :rtype: integer, list
        """
        experience_replay = []
        self.reset_env()
        i=0
        while not self.is_terminal():
            pi = self.agents[player].get_action_probs(pit_mode=True)
            experience_replay.append((np.copy(self.env.board), pi, None))
            #pi = np.array(pi)
            mask = self.legal_actions
            masked_pi = pi
            masked_pi[mask==0] = 0
            # take action with highest action value
            action = np.argmax(masked_pi)
            if i < 2:
                action = np.random.choice(len(self.legal_actions), p=mask/(sum(mask)))
            self.execute_step(action)
            player = int(not player)
            i+=1
        # assign rewards
        for player_i, (observation, pi, _) in list(enumerate(experience_replay))[::-1]:
            r = self.env.reward * (1 if player_i%2 else -1)
            experience_replay[player_i] = (observation, pi, r)
        return experience_replay
    

    def reset_env(self):
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
    

    """def generate_move_hist(self):
        env_copy = Environment(self.config)
        hist = [env_copy.get_board_string()]
        for action in self.action_acc:
            env_copy.execute_step(action)
            hist.append(env_copy.get_board_string())
        return hist"""
    
    
    def get_previous_board_string(self):
        if len(self.string_board_hist) > 1:
            return self.string_board_hist[-2]
        return False

    
    def copy_values_over(self, other_env):
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
        self.board, self.legal_actions, self.terminal, self.reward = self.env.last()
    

    def create_gif(self, name):#, observations):
        #boards = [board for board, _, _ in observations]
        #text = [r for _, _, r in observations]
        env = Environment(self.config)
        env.reset_env()

        boards = []
        text = []
        for a in [None] + self.action_acc:
            text.append(f"p1: {env.env.player}   ")
            if a != None:
                env.execute_step(a)
            boards.append(copy.deepcopy(env.board))
            text[-1] += f"p2: {env.env.player}   terminal: {env.is_terminal()}   r: {env.reward}"

        # Custom colormap: 0, 1, and 2 -> light gray
        colors = [(0.83, 0.71, 0.51), (0.83, 0.71, 0.51), (0.83, 0.71, 0.51)]  # R,G,B
        custom_cmap = ListedColormap(colors)
        # List to store images
        images = []

        for k, board in enumerate(boards):
            fig, ax = plt.subplots(figsize=(5, 5))
            cax = ax.matshow(board, cmap=custom_cmap)

            # Plot lines for the grid
            for i in range(5):
                ax.axhline(i - 0.5, lw=1, color="black", zorder=5)
                ax.axvline(i - 0.5, lw=1, color="black", zorder=5)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot stones
            for i in range(5):
                for j in range(5):
                    if board[i, j] == 1:
                        plt.scatter(j, i, c='black', s=1000, zorder=10)
                    if board[i, j] == -1:
                        plt.scatter(j, i, c='white', s=1000, zorder=10)
            # add text
            fig.suptitle(f"{text[k]}")
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            images.append(imageio.imread(buf))

        # Create GIF
        imageio.mimsave(f'{env.config["images folder"]}{name}.gif', images, duration=0.5)
        plt.close()
        

    def get_board_string(self):
        flattened_board = str(self.env.board.flatten())
        # Create a translation table to remove specified characters
        trans_table = str.maketrans("", "", " .\n[]")
        # Apply the translation to the string
        result = flattened_board.translate(trans_table)
        return result
    

class Gobang_Env:
    def __init__(self, width=19, height=19, required_win_pieces=5):
        self.width, self.height = width, height
        self.board = np.zeros((height, width))
        self.win_pieces = required_win_pieces
        self.player = 0  # alternates between 0 and 1
        self.pieces = {0: 1, 1: -1}  # key=player, value=piece
        
        self.reward = 0
        self.legal_actions = self.get_legal_actions()
        self.terminal = False


    def step(self, action: int):
        # action space is a self.width*self.height large 1 dimensional array
        row = action // self.width
        column = action - (row * self.width)
        # check if action is legal
        if not self.get_legal_actions()[action]:
            print("! Illegal Action:", action, "!")
            print(np.round(self.board))
            print(self.legal_actions.reshape((self.height, self.width)))
            input()
            return False
        # place piece on the board
        self.board[row, column] = self.pieces[self.player]
        # remove enemy pieces if they where caught
        self.remove_enemy(row, column)
        self.terminal = self.is_terminal()
        if self.terminal:
            if self.is_draw():
                self.reward = 0
            elif self.player == 0:
                self.reward = 1
            elif self.player == 1:
                self.reward = -1
        self.player = int(not self.player)
        self.legal_actions = self.get_legal_actions()
        return True


    def last(self):
        return (self.board, self.legal_actions, self.terminal, self.reward)


    def get_legal_actions(self):
        # legal moves (0's represent illegal moves, 1's represent legal moves)
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
        # check in every possible direction if removal condition is met
        player_color = self.pieces[self.player]
        directions = [(1, 0), (-1, 0),
                      (0, 1),(0, -1),
                      (1, 1), (1, -1),
                      (-1, 1), (-1, -1)]
        for ydir, xdir in directions:
            y, x = row + ydir * 3, column + xdir * 3
            if y >= 0 and y < self.height and x >= 0 and x < self.width and self.board[y, x] == player_color:
                pattern = []
                ye1, xe1 = row + ydir * 2, column + xdir * 2
                pattern.append(self.board[ye1, xe1])
                ye2, xe2 = row + ydir * 1, column + xdir * 1
                pattern.append(self.board[ye2, xe2])
                enemy_player_color = self.pieces[int(not self.player)]
                # check if both inner pieces are from the enemy player
                if np.all(np.array(pattern) == enemy_player_color):
                    # remove enemy pieces
                    self.board[ye1, xe1] = 0
                    self.board[ye2, xe2] = 0
    

    def is_draw(self):
        return not np.any(self.board == 0)
    
    
    def is_terminal(self):
        if self.is_draw(): return True

        def check(arr):
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
        
        if check(self.board): return True
        if check(self.board.T): return True
        if check_diagonals(self.board): return True
        if check_diagonals(np.flipud(self.board)): return True
        return False
    

    def copy_env(self):
        new_env = Gobang_Env(self.width, self.height, self.win_pieces)
        new_env.board = np.copy(self.board)
        new_env.player = self.player
        new_env.reward = self.reward
        new_env.legal_actions = new_env.get_legal_actions()
        new_env.terminal = new_env.is_terminal()
        return new_env