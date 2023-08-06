from src.mcts.monte_carlo_tree_search import MCTS

from collections import deque
import numpy as np
from random import shuffle
import tensorflow as tf


class AlphaZero:
    def __init__(self, env, nnet, num_self_play, self_play_deque_length,
                 max_example_number, win_threshold, mcts_config):
        self.env = env
        self.nnet = nnet

        self.mcts_config = mcts_config
        # copy of nnet, used to compete against new network
        self.pnet = nnet.clone_network()
        # path where weights are saved
        self.mcts = MCTS(self.env, self.nnet, self.mcts_config)
        # number of self-play steps
        self.num_self_play = num_self_play
        # max amount of train examples that are saved
        self.max_self_play_examples = self_play_deque_length
        # example history
        self.max_examples = max_example_number
        self.example_hist = []
        # threshold prob at which nnet will be chosen as new network
        self.win_prob_threshold = win_threshold
    
    def train(self, num_iter=1000):
        # repeat self-play-train self.num_iter times
        for i in range(num_iter):
            print("I:", i)
            sp_examples = deque([], maxlen=self.max_self_play_examples)
            # self-play self.num_self_play times
            for spi in range(self.num_self_play):
                print("spi:", spi)
                # reset the monte carlo tree
                self.mcts.reset()
                new_train_examples = self.execute_episode()
                if spi < 10:
                    self.env.create_gif(f"self_play_n{spi}")
                sp_examples += new_train_examples
            # add self play examples to example hist
            self.example_hist.append(sp_examples)
            # remove oldest example if there are to many examples
            while len(self.example_hist) > self.max_examples:
                self.example_hist.pop(0)
            print("Example Hist Len:", len(self.example_hist))
            train_examples = []
            # shuffle examples
            for e in self.example_hist:
                train_examples.extend(e)
            shuffle(train_examples)
            for e in train_examples[:len(train_examples)//3]:
                board, pi, v = e
                r = np.random.rand()
                if r < 0.5: 
                    train_examples.append((np.fliplr(board), pi, v))
                if r < 0.25:
                    train_examples.append((np.flipud(board), pi, v))
                if r > 0.25:
                    train_examples.append((np.flipud(np.fliplr(board)), pi, v))
            shuffle(train_examples)

            # save nnet weights
            self.nnet.save_weights()
            # load weights into pnet
            self.pnet.load_weights()
            
            # train neural network
            self.nnet.train_nnet(train_examples)
            #self.nnet.train(train_examples)

            # create mcts-object for both neural networks
            nmcts = MCTS(self.env, self.nnet, self.mcts_config)
            pmcts = MCTS(self.env, self.pnet, self.mcts_config)
            
            # pit new policies against each other
            n_wins, p_wins, draws = self.env.pit(
                nmcts,
                pmcts
            )
            print(n_wins, p_wins, draws)
            if n_wins + p_wins == 0 or n_wins / (n_wins + p_wins) < self.win_prob_threshold:
                print("Keeping old network.")
                self.nnet.load_weights()
            else:
                print("Saving new network!!")
                self.nnet.save_weights()
                self.nnet.save_model()

    
    def execute_episode(self):
        experience_replay = []
        self.env.reset_env()
        # repeat until game ended
        while not self.env.is_terminal():
            #self.env.render()
            pi = self.mcts.get_action_probs()
            experience_replay.append((self.env.board, pi, None))
            # choose action based on the policy - only legal actions
            action = np.random.choice(len(pi), p=pi)
            self.env.execute_step(action)

        for player_i, (observation, pi, _) in enumerate(experience_replay):
            # flip player reward when looking at opponent
            reward = self.env.reward
            r = reward if player_i%2 else -reward
            experience_replay[player_i] = (observation, pi, r)
        return experience_replay