import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl_glue import RLGlue
from lunar_lander import LunarLanderEnvironment
from agent import BaseAgent
from collections import deque, namedtuple
from tqdm import tqdm
import os
from random import Random

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'terminal', 'next_state'))


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        self.buffer = deque([], maxlen=size)
        self.minibatch_size = minibatch_size
        self.rand_generator = Random(seed)
        self.max_size = size

    def append(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self):
        return self.rand_generator.sample(self.buffer, self.minibatch_size)

    def size(self):
        return len(self.buffer)


def build_action_value_network(network_config):
    state_dim = network_config.get("state_dim")
    num_hidden_units = network_config.get("num_hidden_units")
    num_actions = network_config.get("num_actions")

    initializer = tf.keras.initializers.Orthogonal(
        seed=network_config.get("seed"))

    states = layers.Input(shape=(state_dim,))
    hidden = layers.Dense(num_hidden_units, activation="relu",
                          kernel_initializer=initializer)(states)
    actions = layers.Dense(num_actions, activation="linear",
                           kernel_initializer=initializer)(hidden)

    return keras.Model(inputs=states, outputs=actions)


def softmax(action_values, tau=1.0):
    preferences = None
    max_preference = None

    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)

    reshaped_max_preference = max_preference.reshape((-1, 1))
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences

    action_probs = action_probs.squeeze()
    return action_probs


def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):

    q_next_mat = current_q(next_states)
    probs_mat = softmax(q_next_mat, tau)
    v_next_vec = np.sum(probs_mat * q_next_mat, axis=1) * (1 - terminals)

    target_vec = rewards + discount * v_next_vec

    q_mat = network(states)
    batch_indices = tf.range(q_mat.shape[0])
    full_indices = tf.stack([batch_indices, tf.constant(actions)], axis=1)
    q_vec = tf.gather_nd(q_mat, full_indices)

    return target_vec, q_vec


def optimize_network(experiences, discount, optimizer, network, current_q, tau, loss_function):

    states, actions, rewards, terminals, next_states = map(
        list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)

    with tf.GradientTape() as tape:

        target_vec, q_vec = get_td_error(
            states, next_states, actions, rewards, discount, terminals, network, current_q, tau)

        loss = loss_function(target_vec, q_vec)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self, agent_config):
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = build_action_value_network(
            agent_config['network_config'])
        self.current_q = build_action_value_network(
            agent_config['network_config'])
        self.optimizer = keras.optimizers.Adam(
            **agent_config['optimizer_config'])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        self.loss_function = keras.losses.Huber()

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    def policy(self, state):
        action_values = self.network(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(
            self.num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):

        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.array([state])

        action = self.policy(state)

        self.replay_buffer.append(
            self.last_state, self.last_action, reward, 0, state)

        if self.replay_buffer.size() >= self.replay_buffer.minibatch_size:
            self.current_q.set_weights(self.network.get_weights())
            for _ in range(self.num_replay):

                experiences = self.replay_buffer.sample()

                optimize_network(experiences, self.discount, self.optimizer,
                                 self.network, self.current_q, self.tau, self.loss_function)

        self.last_state = state
        self.last_action = action

        return action

    def agent_end(self, reward):
        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.zeros_like(self.last_state)

        self.replay_buffer.append(
            self.last_state, self.last_action, reward, 1, state)

        if self.replay_buffer.size() >= self.replay_buffer.minibatch_size:
            self.current_q.set_weights(self.network.get_weights())
            for _ in range(self.num_replay):

                experiences = self.replay_buffer.sample()

                optimize_network(experiences, self.discount, self.optimizer,
                                 self.network, self.current_q, self.tau, self.loss_function)

        # self.network.save("rl_agent_q_function_model");

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")


def run_experiment(environment, agent, agent_parameters, experiment_parameters):

    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                                 experiment_parameters["num_episodes"]))

    env_info = {}

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)

        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])

            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)


if __name__ == '__main__':
    experiment_parameters = {
        "num_runs": 1,
        "num_episodes": 200,
        "timeout": 500
    }

    current_env = LunarLanderEnvironment

    agent_parameters = {
        'network_config': {
            'state_dim': 8,
            'num_hidden_units': 256,
            'num_actions': 4
        },
        'optimizer_config': {
            'learning_rate': 1e-3,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8
        },
        'replay_buffer_size': 50000,
        'minibatch_sz': 8,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.001
    }
    current_agent = Agent

    run_experiment(current_env, current_agent,
                   agent_parameters, experiment_parameters)
