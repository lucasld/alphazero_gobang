from src.mcts.monte_carlo_tree_search import MCTS
from src.utils import tools
from collections import deque
import numpy as np
from random import shuffle
import tensorflow as tf
import pickle
import os
from time import time



class AlphaZero:
    def __init__(self, env, nnet, alphazero_config, mcts_config):
        self.env = env
        self.nnet = nnet

        self.alphazero_config = alphazero_config

        self.mcts_config = mcts_config
        # copy of nnet, used to compete against new network
        self.pnet = nnet.clone_network()
        # path where weights are saved
        self.mcts = MCTS(self.env, self.nnet, self.mcts_config)
        # number of self-play steps
        self.num_self_play = alphazero_config["number self-play"]
        # max amount of train examples that are saved
        self.max_self_play_examples = alphazero_config["self play deque length"]
        # example history
        self.max_examples = alphazero_config["max example number"]

        self.exp_path = self.alphazero_config["experience path"]
        self.example_hist = tools.load_example_hist(self.exp_path)

        # threshold prob at which nnet will be chosen as new network
        self.win_prob_threshold = alphazero_config["pit win threshold"]
    
    def train(self, num_iter=1000):
        # repeat self-play-train self.num_iter times
        for i in range(num_iter):
            self.example_hist = tools.load_example_hist(self.exp_path)
            print("Iteration:", i)
            sp_examples = deque([], maxlen=self.max_self_play_examples)
            # self-play self.num_self_play times
            self.mcts.reset()
            for spi in range(self.num_self_play):
                print(f"self play number: {spi}/{self.num_self_play}")
                new_train_examples = self.execute_episode()
                if spi < 3:
                    self.env.create_gif(f"self_play_n{spi}")#, new_train_examples)
                sp_examples += new_train_examples
            # add self play examples to example hist
            self.example_hist.append(sp_examples)

            # remove oldest example if there are to many examples
            while len(self.example_hist) > self.max_examples:
                self.example_hist.pop(0)

            # saving example history
            tools.save_example_hist(self.exp_path, self.example_hist)
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
            new_network_has_won = self.env.pit(nmcts, pmcts, self.win_prob_threshold)
            if new_network_has_won:
                print("Saving new network!!")
                self.nnet.save_weights()
                self.nnet.save_model()
            else:
                print("Keeping old network.")
                self.nnet.load_weights()
            """
            if n_wins + p_wins == 0 or n_wins / (n_wins + p_wins) < self.win_prob_threshold:
                print("Keeping old network.")
                self.nnet.load_weights()
            else:
                print("Saving new network!!")
                self.nnet.save_weights()
                self.nnet.save_model()
            """

    
    def execute_episode(self):
        experience_replay = []
        self.env.reset_env()
        #self.mcts.reset()
        # repeat until game ended
        times = [time()]
        #pi = self.mcts.get_action_probs()
        #experience_replay.append((np.copy(self.env.board), pi, None))
        while not self.env.is_terminal():
            #self.env.render()
            # choose action based on the policy - only legal actions
            pi = self.mcts.get_action_probs()  #TODO
            experience_replay.append((np.copy(self.env.board), pi, None))
            # choose and perform action
            action = np.random.choice(len(pi), p=pi)
            self.env.execute_step(action)
            # timing
            times.append(time())
            mean_time = np.round(np.mean((np.array(times[1:]) - np.array(times[:-1]))[:-1]), 3)
            print("\raverage move time:", mean_time, end="")
        # assign rewards
        for player_i, (observation, pi, _) in list(enumerate(experience_replay))[::-1]:
            r = self.env.reward * (1 if player_i%2 else -1)
            experience_replay[player_i] = (observation, pi, r)
        print()
        return experience_replay