import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

def softmax(action_values, tau=0.001):
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

def policy(state,network,tau=0.001):
    action_values = network(state)
    probs_batch = softmax(action_values, tau)
    action = np.random.RandomState(0).choice(4, p=probs_batch.squeeze())
    return action

if __name__ == "__main__":
    model = keras.models.load_model('rl_agent_q_function_model')
    trials_no = 1
    for i in range(10,10+trials_no):

        env = gym.make("LunarLander-v2")
        env.seed(i)

        reward = 0.0
        observation = env.reset()
        is_terminal = False
        reward_obs_term = (reward, observation, is_terminal)

        while(not is_terminal):
            last_state = reward_obs_term[1]
            state = np.array([last_state])
            action = policy(state,model)
            current_state, reward, is_terminal, _ = env.step(action)
            
            reward_obs_term = (reward, current_state, is_terminal)

            env.render()