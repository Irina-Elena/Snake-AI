from buffer import Buffer, build_dqn, load_game
import numpy as np


class Agent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_shape, epsilon_dec, epsilon_end,
                 memory_size, file_name, activations):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.experiences = Buffer(memory_size, input_shape, n_actions)

        self.q_eval = build_dqn(alpha, n_actions, input_shape, 100, 100, activations)
        self.file = file_name

    def add_experience(self, state, action, reward, new_state):
        self.experiences.store_transition(state, new_state, reward, action)

    def choose_action(self, state):
        state = np.array([state])
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # print(state)
            actions = self.q_eval.predict(state)
            # print(actions)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.experiences.contor > self.batch_size:
            state, action, reward, new_state = self.experiences.get_batch(self.batch_size)
            actions = np.dot(action, self.action_space)
            target = self.q_eval.predict(state)
            new_values = self.q_eval.predict(new_state)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            target[batch_index, actions] = reward + self.gamma * np.max(new_values, axis=1)

            self.q_eval.fit(state, target, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

    def save_game(self):
        self.q_eval.save(str(self.file))

    def load_game(self):
        self.q_eval = load_game(self.file)
