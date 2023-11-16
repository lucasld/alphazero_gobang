import pickle
import os
import json
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import imageio
from src.environment import gobang


def load_example_hist(path):
    """Load the example history from a specified path.

    :param path: The path to the example history file
    :type path: str
    :return: Loaded example history or an empty list
    :rtype: list
    """
    if os.path.exists(path):
        path = f"{path}data"
        if os.path.exists(path):
            # Unpickling
            with open(path, "rb") as fp:
                example_hist = pickle.load(fp)
            return example_hist
    return []
    

def save_example_hist(dir_path, example_hist):
    """Save the example history to a specified directory.

    :param dir_path: The path to directory fpr saving
    :type dir_path: str
    :param example_hist: The example history to be saved
    :type example_hist: list
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = f"{dir_path}data"
    # Pickling
    with open(path, "wb") as fp:
        pickle.dump(example_hist, fp)


def add_new_examples(dir_path, new_examples):
    """Add new examples to the existing example history.

    :param dir_path: Path to the directory where the example history is stored
    :type dir_path: str
    :param new_examples: The new examples to be added to the history.
    :type new_examples: list
    """
    example_hist = load_example_hist(dir_path)
    if example_hist:
        example_hist[-1] += new_examples
    else:
        example_hist = [new_examples]
    save_example_hist(dir_path, example_hist)


def plot_training_history(history, save_path="training_plot.png"):
    """Plot the training and validation loss, as well as accuracy, over epochs.

    :param history: history dictionary containing loss and accuracy values
    :type history: dict
    :param save_path: path to save the plot image, default is "training_plot.png"
    :type save_path: str
    """
    plt.figure(figsize=(16, 8))
    # Add red vertical lines at all the epochs where training started
    def draw_lines():
        for epoch in history['epochs']:
            plt.axvline(x=epoch-1, color='red', linestyle='--', linewidth=0.5)
    
    # Plot for 'loss'
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    draw_lines()
    plt.legend(['Total Loss'], loc='upper right')

    # Plot for 'policy_output_loss'
    plt.subplot(2, 2, 2)
    plt.plot(history['policy_output_loss'])
    plt.title('Policy Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    draw_lines()
    plt.legend(['Policy Output Loss'], loc='upper right')

    # Plot for 'value_output_loss'
    plt.subplot(2, 2, 3)
    plt.plot(history['value_output_loss'])
    plt.title('Value Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    draw_lines()
    plt.legend(['Value Output Loss'], loc='upper right')

    # Plot for 'policy_output_accuracy'
    plt.subplot(2, 2, 4)
    plt.plot(history['policy_output_accuracy'])
    plt.title('Policy Output Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    draw_lines()
    plt.legend(['Policy Output Accuracy'], loc='lower right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig(save_path)


def add_and_plot_history(path, history=False):
    """Plot the complete training history. Loads the exisiting history,
    optionally adding new history, saving and plotting it.
    
    :param path: path to the history file
    :type path: string
    :param history: new history to add to existing history
    :type history: dictonary
    """
    # Check if history file already exists
    if os.path.exists(path):
        with open(path, 'r') as file:
            existing_history = json.load(file)
        # If new history is provided, update the existing history
        if history:
            if 'epochs' not in existing_history.keys():
                existing_history['epochs'] = []
            existing_history['epochs'].append(len(existing_history['loss']))
            for key in existing_history.keys():
                if key != 'epochs':
                    existing_history[key].extend(history.history[key])
    else:
        # If the history file doesn't exist, use the provided history
        if history:
            existing_history = {
                'loss': [], 'policy_output_loss': [],
                'value_output_loss': [], 'policy_output_accuracy': []
            }
            for key in existing_history.keys():
                existing_history[key].extend(history.history[key])
            existing_history['epochs'] = [len(existing_history['loss'])]
        else:
            existing_history = {
                'loss': [], 'policy_output_loss': [],
                'value_output_loss': [], 'policy_output_accuracy': [],
                'epochs': []
            }
    # Save the combined history back to the file
    with open(path, 'w') as file:
        json.dump(existing_history, file)
    # Plot the complete training history
    plot_training_history(existing_history, path[:-4]+"png")


def create_gif(name, action_hist, env_config):
        """Creates a GIF representing the sequence of game boards during 
        a match.

        :param name: Name of the GIF file.
        :type name: str
        :param action_hist: sequence of actions taken in the game
        :type action_hist: list of ints
        :param env_config: configuration parameters for an environment
        :type env_config: dict
        """
        env = gobang.Environment(env_config)
        env.reset_env()
        boards = []
        text = []
        # Iterate over actions to create the sequence
        for a in [None] + action_hist:
            text.append(f"player before: {env.env.player}   ")
            # Execute the action and update the environment
            if a is not None:
                env.execute_step(a)
            # Collect board state and associated information
            boards.append(copy.deepcopy(env.board))
            text[-1] += f"player now: {env.env.player}   "
            text[-1] += f"terminal: {env.is_terminal()}   r: {env.reward}"

        # Custom colormap: 0, 1, and 2 -> light gray
        colors = [(0.83, 0.71, 0.51), (0.83, 0.71, 0.51), (0.83, 0.71, 0.51)]
        custom_cmap = ListedColormap(colors)
        # List to store images
        images = []

        for k, board in enumerate(boards):
            fig, ax = plt.subplots(figsize=(env_config['width'],
                                            env_config['height']))
            cax = ax.matshow(board, cmap=custom_cmap)

            # Plot lines for the grid
            for i in range(5):
                ax.axhline(i - 0.5, lw=1, color="black", zorder=5)
                ax.axvline(i - 0.5, lw=1, color="black", zorder=5)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot stones
            for i in range(env_config['width']):
                for j in range(env_config['height']):
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


def visualize_board(env):
    """Visualizes the current game board in a human-readable format.

    This function prints a textual representation of the game board using 'X'
    for Player -1, 'O' for Player 1, and empty space for unoccupied positions.

    :param env: Takes GobangEnvironment as input, not just Environment
    :type env: GobangEnvironment object
    """
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