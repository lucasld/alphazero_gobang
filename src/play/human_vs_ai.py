import numpy as np
from src.utils import tools


def play_game(env, mcts, human_first_move=True, alphazero_config=None):
    replay = []
    player = "human" if human_first_move else "ai"
    env.reset_env()
    # repeat until game ended
    while not env.is_terminal():
        tools.visualize_board(env)
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
    tools.visualize_board(env)
    
    # assign rewards
    for player_i, (observation, pi, _) in list(enumerate(replay))[::-1]:
        r = env.reward * (1 if player_i%2 else -1)
        replay[player_i] = (observation, pi, r)
    
    # save replay
    if alphazero_config:
        exp_path = alphazero_config["experience path"]
        tools.add_new_examples(exp_path, replay)