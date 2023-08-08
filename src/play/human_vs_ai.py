import numpy as np


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

def play_game(env, mcts, human_first_move=True):
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
            player = "human"
        elif player == "human":
            r, c = input(":").split(" ")
            r, c = int(r), int(c)
            action = r * env.width + c
            while not env.execute_step(action):
                r, c = input(":").split(" ")
                r, c = int(r), int(c)
                action = r * env.width + c
            player = "ai"
    visualize_board(env)
            