import argparse
import os

from src.training.trainer import AlphaZero
from src.environment.gobang import Environment
from src.model.neural_network import NeuralNetwork
from src.play.human_vs_ai import play_game
from config import TRAINING_CONFIG, PLAYING_CONFIG

def train_model(config):
    model_name = config["name"]
    weight_path = "./src/model/model_weights/" + model_name
    input_shape = (config["game"]["height"], config["game"]["width"], 1)
    policy_shape = input_shape[0] * input_shape[1]
    load_weights_path = False
    if os.path.exists(weight_path + '.keras'):
        print("Preexisting model")
        load_weights_path = weight_path
    num_iterations = config["alpha zero"]["number iterations"]
    win_threshold = config["alpha zero"]["pit win threshold"]
    num_self_play = config["alpha zero"]["number self-play"]
    deque_length = config["alpha zero"]["self play deque length"]
    num_examples_max = config["alpha zero"]["max example number"]
    
    # initialize the environment object
    env = Environment(config["game"])
    # initialize the nnet managing object
    nnet = NeuralNetwork(weight_path,
                         input_shape=input_shape,
                         policy_shape=policy_shape,
                         load_weights_path=load_weights_path)
    # initialize the alpha zero object
    trainer = AlphaZero(env, nnet, num_self_play,
                        deque_length, num_examples_max, win_threshold,
                        config["mcts"])
    # Train the model using the specified configuration
    trainer.train(num_iter=num_iterations)

def play_human_vs_ai(config):
    # Play the game against the trained AI using the specified configuration
    play_game(config)

def main():
    parser = argparse.ArgumentParser(description="Alpha Zero Gobang")
    parser.add_argument("--train", action="store_true", help="Train the Alpha Zero model")
    parser.add_argument("--play", action="store_true", help="Play against the trained Alpha Zero model")

    args = parser.parse_args()

    if args.train:
        print("Training the Alpha Zero model...")
        train_model(TRAINING_CONFIG)

    elif args.play:
        print("Playing against the trained Alpha Zero model...")
        play_human_vs_ai(PLAYING_CONFIG)

    else:
        print("Please specify either --train or --play.")

if __name__ == "__main__":
    main()
