import numpy as np
import os
import pickle

def visualize_board(env):
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


def play_game(env, mcts, human_first_move=True, alphazero_config=None):
    replay = []
    player = "human" if human_first_move else "ai"
    env.reset_env()
    # repeat until game ended
    while not env.is_terminal():
        visualize_board(env)
        if player == "ai":
            pi = mcts.get_action_probs()
            # choose action based on the policy - only legal actions
            action = np.argmax(pi)
            env.execute_step(action)
            # switching player
            player = "human"
        elif player == "human":
            r, c = input(":").split(" ")
            r, c = int(r), int(c)
            action = r * env.width + c
            while not env.execute_step(action):
                r, c = input(":").split(" ")
                r, c = int(r), int(c)
                action = r * env.width + c
            pi = np.zeros(env.action_space_size)
            pi[action] = 1.0
            # switching player
            player = "ai"
        replay.append((env.board, pi, None))
    visualize_board(env)
    
    # assign rewards
    for player_i, (observation, pi, _) in list(enumerate(replay))[::-1]:
        r = env.reward * (1 if player_i%2 else -1)
        replay[player_i] = (observation, pi, r)
    
    # save replay
    if alphazero_config:
        history = load_example_hist(alphazero_config)
        history.append(replay)
        save_example_hist(alphazero_config, history)


def load_example_hist(alphazero_config):
    if "experience path" in alphazero_config.keys():
        exp_path = alphazero_config["experience path"]
        if os.path.exists(exp_path):
            path = f"{exp_path}data"
            if os.path.exists(path):
                # Unpickling
                with open(path, "rb") as fp:
                    example_hist = pickle.load(fp)
                print("previous experience loaded!")
                return example_hist
        else:
            os.makedirs(exp_path)
            print("created directory for saving experience later..")
    

def save_example_hist(alphazero_config, example_hist):
    if "experience path" in alphazero_config.keys():
        print("saving example history...")
        path = f"{alphazero_config['experience path']}data"
        with open(path, "wb") as fp:   #Pickling
            pickle.dump(example_hist, fp)
            