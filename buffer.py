from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
import numpy as np


class Buffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.memory_size = max_size
        self.contor = 0
        self.state_memory = np.zeros((self.memory_size, input_shape))
        self.new_states = np.zeros((self.memory_size, input_shape))
        self.actions = np.zeros((self.memory_size, n_actions), dtype=np.int8)
        self.rewards = np.zeros(self.memory_size)

    def store_transition(self, state, new_state, reward, action):
        index = self.contor % self.memory_size
        self.state_memory[index] = state
        self.new_states[index] = new_state
        self.actions[index] = np.zeros(self.actions.shape[1])
        self.actions[index][action] = 1
        self.rewards[index] = reward
        self.contor += 1

    def get_batch(self, batch_size):
        max_elements_batch = min(self.contor, self.memory_size)
        batch = np.random.choice(max_elements_batch, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]

        return states, actions, rewards, new_states


def build_dqn(learning_rate, output_size, input_shape, layer1_size, layer2_size, activations):
    model = Sequential([
        Dense(layer1_size, input_shape=(input_shape,)),
        Activation(str(activations[0])),
        Dense(layer2_size),
        Activation(str(activations[1])),
        Dense(output_size), Activation('softmax')])

    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mse')

    return model


def load_game(filename):
    return load_model(str(filename), compile=True)
