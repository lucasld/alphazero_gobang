import argparse

from config import TRAINING_CONFIG, PLAYING_CONFIG
from src.environment.gobang import Environment
from src.mcts.monte_carlo_tree_search import MCTS
from src.model.neural_network import NeuralNetwork
from src.play.human_vs_ai import play_game
from src.training.trainer import AlphaZero


def train_model(config: dict) -> None:
    """Train the AlphaZero model.

    :param config: the dictionary containing all configuration parameters for
        training.
    :type config: dictionary
    """
    # initialize the environment object
    env = Environment(config["game"])
    # initialize the nnet managing object
    nnet = NeuralNetwork(config, load_existing_model=True)
    # initialize the alpha zero object
    trainer = AlphaZero(env, nnet, config["alpha zero"], config["mcts"])

    # Train the model using the specified configuration
    num_iterations = config["alpha zero"]["number iterations"]
    trainer.train(num_iter=num_iterations)


def play_human_vs_ai(config: dict) -> None:
    """Play against the trained Alpha Zero model.
    
    :param config: the dictionary containing all configutation to play a game
    :type config: dictionary
    """
    # initialize the environment object
    env = Environment(config["game"])
    # initialize the nnet managing object
    nnet = NeuralNetwork(config)
    # intialize mcts object
    mcts = MCTS(env, nnet, config=config["mcts"])

    # Play the game against the trained AI using the specified configuration
    # add alphazero_config if game should be added to the experience replay
    play_game(env, mcts, human_first_move=False, alphazero_config=config["alpha zero"])


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
