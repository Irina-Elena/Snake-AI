from ple.games.snake import Snake
from ple import PLE
import numpy as np
from agent import Agent
import pygame
import sys
import matplotlib.pyplot as plt


def get_dist(head_x, head_y, obs_x, obs_y):
    return ((head_x - obs_x) ** 2 + (head_y - obs_y) ** 2) ** 0.5


def get_state(state):
    head_x, head_y = state[0], state[1]
    min_dist_walls = min(get_dist(head_x, head_y, head_x, 0), get_dist(head_x, head_y, 0, head_y),
                         get_dist(head_x, head_y, 600, head_y), get_dist(head_x, head_y, head_x, 600))
    return [state[0], state[1], state[2], state[3], min(min(state[4][4:]), min_dist_walls)]


def vision(state, direction):
    my_vision = [[0, 0] for _ in range(3)]
    head_x, head_y = state[0], state[1]
    food_x, food_y = state[2], state[3]

    # food

    if direction == "Left":
        if head_x - food_x >= 0:
            my_vision[2][0] = 1
        if food_y - head_y < 0:
            my_vision[0][0] = 1
        else:
            my_vision[1][0] = 1

        # wall
        if head_x <= 50:
            my_vision[2][1] = -1
        if 600 - head_y <= 50:
            my_vision[1][1] = -1
        if head_y <= 50:
            my_vision[0][1] = -1

        # body
        for body_x, body_y in state[5][3:]:
            # print(body_x,body_y)
            if head_x - body_x >= 0:
                my_vision[2][1] = -1
            if body_y - head_y < 0:
                my_vision[0][1] = -1
            else:
                my_vision[1][1] = -1

    elif direction == "Up":
        if head_y - food_y >= 0:
            my_vision[2][0] = 1
        if head_x - food_x < 0:
            my_vision[0][0] = 1
        else:
            my_vision[1][0] = 1

        # wall
        if head_y <= 50:
            my_vision[2][1] = -1
        if 600 - head_x <= 50:
            my_vision[0][1] = -1
        if head_x <= 50:
            my_vision[1][1] = -1

        # body
        for body_x, body_y in state[5][3:]:
            # print(body_x,body_y)
            if head_y - body_y >= 0:
                my_vision[2][1] = -1
            if body_x - head_x < 0:
                my_vision[0][1] = -1
            else:
                my_vision[1][1] = -1

    elif direction == "Right":
        if head_x - food_x <= 0:
            my_vision[2][0] = 1
        if food_y - head_y >= 0:
            my_vision[0][0] = 1
        else:
            my_vision[1][0] = 1

        # wall
        if 600 - head_x <= 50:
            my_vision[2][1] = -1
        if head_y <= 50:
            my_vision[1][1] = -1
        if 600 - head_y <= 50:
            my_vision[0][1] = -1

        # body
        for body_x, body_y in state[5][3:]:
            if head_x - body_x <= 0:
                my_vision[2][1] = -1
            if body_y - head_y >= 0:
                my_vision[0][1] = -1
            else:
                my_vision[1][1] = -1

    else:
        if head_y - food_y <= 0:
            my_vision[2][0] = 1
        if head_x - food_x >= 0:
            my_vision[0][0] = 1
        else:
            my_vision[1][0] = 1

        # wall
        if 600 - head_y <= 50:
            my_vision[2][1] = -1
        if head_x <= 50:
            my_vision[0][1] = -1
        if 600 - head_x <= 50:
            my_vision[1][1] = -1

        # body
        for body_x, body_y in state[5][3:]:
            if head_y - body_y <= 0:
                my_vision[2][1] = -1
            if body_x - head_x >= 0:
                my_vision[0][1] = -1
            else:
                my_vision[1][1] = -1

    output = []

    [output.extend(item) for item in my_vision]
    return output


def prepare_corect_directions(direction):
    direction = str(direction)
    if direction == "Left":
        return {119: "Up", 115: "Down", 97: "Left"}
    if direction == "Right":
        return {115: "Down", 119: "Up", 100: "Right"}
    if direction == "Up":
        return {100: "Right", 97: "Left", 119: "Up"}
    if direction == "Down":
        return {97: "Left", 100: "Right", 115: "Down"}


def process_state(state):
    return np.array([state.values()])


def test():
    game = Snake(600, 600)
    p = PLE(game, fps=60, state_preprocessor=process_state, force_fps=True, display_screen=True,frame_skip = 2,
            reward_values={"positive": 100.0,
                           "negative": -50.0,
                           "tick": -0.1,
                           "loss": -70.0,
                           "win": 5.0})
    agent = Agent(alpha=float(sys.argv[1]), gamma=float(sys.argv[2]), n_actions=3, epsilon=0.01, batch_size=100,
                  input_shape=6, epsilon_dec=0.99999,
                  epsilon_end=0.001,
                  memory_size=500000, file_name=sys.argv[3], activations=[str(sys.argv[4]), str(sys.argv[5])])
    p.init()
    agent.load_game()
    scores = []

    for _ in range(200):
        if p.game_over():
            p.reset_game()
        apples = 0
        initial_direction = "Right"
        while not p.game_over():
            old_state = np.array(vision(list(p.getGameState()[0]), initial_direction))

            action = agent.choose_action(old_state)
            possible_directions = prepare_corect_directions(initial_direction)
            possible_directions_tuples = list(zip(possible_directions.keys(), possible_directions.values()))
            direction = possible_directions_tuples[action]
            initial_direction = direction[1]

            reward = p.act(direction[0])
            if reward > 50.0:
                apples += reward

        scores.append(apples)
    return scores


def train():
    game = Snake(600, 600)
    p = PLE(game, fps=60, state_preprocessor=process_state, force_fps=True, display_screen=False, frame_skip=2,
            reward_values={"positive": 100.0,
                           "negative": -50.0,
                           "tick": -0.1,
                           "loss": -110.0,
                           "win": 5.0})
    agent = Agent(alpha=float(sys.argv[1]), gamma=float(sys.argv[2]), n_actions=3, epsilon=0.99, batch_size=100,
                  input_shape=6, epsilon_dec=0.99999,
                  epsilon_end=0.001,
                  memory_size=500000, file_name=sys.argv[3], activations=[str(sys.argv[4]), str(sys.argv[5])])
    p.init()
    # agent.load_game()

    scores = []

    for _ in range(100000):
        if p.game_over():
            p.reset_game()
        score = 0
        initial_direction = "Right"

        while not p.game_over():
            old_state = np.array(vision(list(p.getGameState()[0]), initial_direction))

            action = agent.choose_action(old_state)

            possible_directions = prepare_corect_directions(initial_direction)
            possible_directions_tuples = list(zip(possible_directions.keys(), possible_directions.values()))
            direction = possible_directions_tuples[action]
            initial_direction = direction[1]

            reward = p.act(direction[0])

            new_state = np.array(vision(list(p.getGameState()[0]), initial_direction))
            agent.add_experience(old_state, action, reward, new_state)
            agent.learn()
            score = p.score()
        scores.append(score)
        print(
            f"Score for model iteration number _ {str(sys.argv[3])} with learning_rate {sys.argv[1]}, gama {sys.argv[2]}, activations: {sys.argv[4], sys.argv[5]} is score {score}. Epsilon is {agent.epsilon}")
        agent.save_game()


if __name__ == '__main__':
    if sys.argv[6] == "test":
        scores = test()
        scores = list(map(lambda x:x//100,scores))
        print(max(scores))
        plt.plot(scores)
        plt.savefig(sys.argv[3] + ".png")

    else:
        train()
