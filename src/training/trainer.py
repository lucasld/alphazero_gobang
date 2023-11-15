from src.mcts.monte_carlo_tree_search import MCTS
from src.utils import tools

from collections import deque
import numpy as np
from time import time


class AlphaZero:
    def __init__(self, env, nnet, alphazero_config, mcts_config):
        """Initializes the AlphaZero agent.

        :param env: The environment for the game
        :type env: Environment object
        :param nnet: The neural network manager
        :type nnet: NeuralNetwork object
        :param alphazero_config: Configuration parameters for AlphaZero.
        :type alphazero_config: dict
        :param mcts_config: Configuration parameters for MCTS
        :type mcts_config: dict
        """
        self.env = env
        self.nnet = nnet

        self.alphazero_config = alphazero_config

        self.mcts_config = mcts_config
        # Copy of nnet, used to compete against new network
        self.pnet = nnet.clone_network()
        # Path where weights are saved
        self.mcts = MCTS(self.env, self.nnet, self.mcts_config)
        # Number of self-play steps
        self.num_self_play = alphazero_config["number self-play"]
        # Max amount of train examples that are saved
        self.max_self_play_examples = alphazero_config["self play deque length"]
        # Example history
        self.max_examples = alphazero_config["max example number"]

        self.exp_path = self.alphazero_config["experience path"]
        self.example_hist = tools.load_example_hist(self.exp_path)

        # Threshold prob at which nnet will be chosen as new network
        self.win_prob_threshold = alphazero_config["pit win threshold"]
    

    def train(self, num_iter=1000):
        """Trains the AlphaZero agent through self-play and
        neural network updates.

        :param num_iter: Number of training iterations, defaults to 1000
        :type num_iter: int, optional
        """
        # Number of times new nnet lost against old nnet since last win
        no_update_since = 1
        # Repeat self-play-train self.num_iter times
        for i in range(num_iter):
            self.example_hist = tools.load_example_hist(self.exp_path)
            print("Iteration:", i)
            sp_examples = deque([], maxlen=self.max_self_play_examples)
            for selp_play_i in range(self.num_self_play):
                self.mcts.reset()
                print(f"self play number: {selp_play_i+1}/{self.num_self_play}")
                new_train_examples = self.execute_episode()
                if selp_play_i < 3:
                    tools.create_gif(f"self_play_n{selp_play_i}",
                                     self.env.action_acc,
                                     self.env.config)
                sp_examples += new_train_examples

            # Add self play examples to example hist
            if len(sp_examples):
                self.example_hist.append(sp_examples)

            # Remove oldest example if there are to many examples
            print(len(self.example_hist))
            while len(self.example_hist) > self.max_examples:
                self.example_hist.pop(0)

            # Saving example history
            tools.save_example_hist(self.exp_path, self.example_hist)
            print("Example Hist Len:", len(self.example_hist))
            # Collect all train examples
            train_examples = []
            for e in self.example_hist:
                train_examples.extend(e)
            # Save nnet weights
            self.nnet.save_weights()
            # Load weights into pnet
            self.pnet.load_weights()
            
            # Train neural network
            self.nnet.train_nnet(train_examples)

            # Create mcts-object for both neural networks
            nmcts = MCTS(self.env, self.nnet, self.mcts_config)
            pmcts = MCTS(self.env, self.pnet, self.mcts_config)
            
            # Pit new policies against each other
            new_network_has_won = self.env.pit(nmcts, pmcts, self.win_prob_threshold)
            if new_network_has_won:
                print("Saving new network!!")
                self.nnet.save_weights()
                self.nnet.save_model()
                no_update_since = 1
            else:
                print("Keeping old network.")
                self.nnet.load_weights()
                no_update_since += 1

    
    def execute_episode(self):
        """Executes a single episode of self-play.

        :return: Experience replay data for training
        :rtype: list
        """
        # List to collect experience
        experience_replay = []
        self.env.reset_env()
        times = [time()]
        # Repeat until game terminated
        while not self.env.is_terminal():
            # Choose action based on the policy - only legal actions
            pi = self.mcts.get_action_probs()  #TODO
            experience_replay.append((self.env.get_neutral_board(), pi, None))
            # Choose and perform action
            action = np.random.choice(len(pi), p=pi)
            self.env.execute_step(action)
            # Timing
            times.append(time())
            mean_t = np.mean((np.array(times[1:]) - np.array(times[:-1]))[:-1])
            mean_t = np.round(mean_t, 3)
            print("\raverage move time:", mean_t, end="")
        # Assign rewards
        experience_flipped = list(enumerate(experience_replay))[::-1]
        for player_i, (observation, pi, _) in experience_flipped:
            r = self.env.reward * (1 if player_i%2 else -1)
            experience_replay[player_i] = (observation, pi, r)
        print("\n-")
        return experience_replay