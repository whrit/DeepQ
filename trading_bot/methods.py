import os
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .utils import format_currency, format_position
from .ops import get_state

@tf.function(reduce_retracing=True)
def tf_get_state(data, t, window_size):
    state = get_state(data, t, window_size + 1)  # Add 1 to window_size to get the correct number of differences
    return tf.ensure_shape(state, (window_size,))  # Ensure the shape is correct

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10, checkpoint_interval=5):
    total_profit = 0.0
    data_length = len(data) - 1
    agent.inventory = []
    avg_loss = []
    
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    state = tf_get_state(data_tf, 0, window_size)
    
    for t in tqdm(range(data_length), total=data_length, leave=True, desc=f'Episode {episode}/{ep_count}'):
        reward = 0.0
        next_state = tf_get_state(data_tf, t + 1, window_size)
        
        action = agent.act(tf.expand_dims(state, 0))
        
        if action == 1:  # BUY
            agent.inventory.append(data[t])
        elif action == 2 and agent.inventory:  # SELL
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta
        
        done = (t == data_length - 1)
        agent.remember(state.numpy(), int(action), reward, next_state.numpy(), done)
        
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)
        
        state = next_state
        
        # Save checkpoint
        if (t + 1) % (data_length // checkpoint_interval) == 0:
            checkpoint_episode = f"{episode}_{(t + 1) / data_length:.2f}"
            agent.save(checkpoint_episode)
            print(f"Checkpoint saved at step {t + 1}")

    # Save at the end of the episode
    agent.save(f"{episode}_final")
    
    return (episode, ep_count, total_profit, tf.reduce_mean(avg_loss).numpy())

def evaluate_model(agent, data, window_size, debug):
    total_profit = 0.0
    data_length = len(data) - 1
    history = []
    agent.inventory = []
    
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    state = tf_get_state(data_tf, 0, window_size)  # Remove the +1 here
    
    for t in range(data_length):
        reward = 0.0
        next_state = tf_get_state(data_tf, t + 1, window_size)  # Remove the +1 here as well
        
        action = agent.act(tf.expand_dims(state, 0), is_eval=True)
        
        if action == 1:  # BUY
            agent.inventory.append(data[t])
            history.append((data[t], "BUY"))
            if debug:
                logging.debug(f"Buy at: {format_currency(data[t])}")
        
        elif action == 2 and agent.inventory:  # SELL
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta
            history.append((data[t], "SELL"))
            if debug:
                logging.debug(f"Sell at: {format_currency(data[t])} | Position: {format_position(delta)}")
        
        else:  # HOLD
            history.append((data[t], "HOLD"))
        
        done = (t == data_length - 1)
        agent.memory.append((state.numpy(), int(action), reward, next_state.numpy(), done))
        state = next_state
        
        if done:
            return total_profit, history