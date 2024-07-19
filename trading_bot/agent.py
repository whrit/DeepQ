import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

class Agent:
    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy
        self.state_size = state_size
        self.action_size = 3
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(128, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(self.action_size)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray) and state.ndim == 2:
            state = state.flatten()
        if isinstance(next_state, np.ndarray) and next_state.ndim == 2:
            next_state = next_state.flatten()
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def act(self, state, is_eval=False):
        state = tf.squeeze(state)  # Remove extra dimensions
        if state.shape.ndims == 1:
            state = tf.expand_dims(state, 0)  # Add batch dimension if needed
        
        if not is_eval and tf.random.uniform(()) <= self.epsilon:
            return tf.random.uniform((), minval=0, maxval=self.action_size, dtype=tf.int32)
        
        if self.first_iter:
            self.first_iter = False
            return tf.constant(1, dtype=tf.int32)

        action_probs = self.model(state)
        return tf.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([state for state, _, _, _, _ in mini_batch])
        actions = np.array([action for _, action, _, _, _ in mini_batch])
        rewards = np.array([reward for _, _, reward, _, _ in mini_batch])
        next_states = np.array([next_state for _, _, _, next_state, _ in mini_batch])
        dones = np.array([done for _, _, _, _, done in mini_batch])

        # Debug logging
        print(f"States shape: {states.shape}")
        print(f"Next states shape: {next_states.shape}")
        print(f"Expected state size: {self.state_size}")

        # Determine the actual state size
        actual_state_size = states.shape[-1]

        # Reshape states and next_states to match the model's input shape
        states = states.reshape(batch_size, actual_state_size)
        next_states = next_states.reshape(batch_size, actual_state_size)

        # Convert to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # Compute Q-values for current states
        current_q_values = self.model(states)

        # Compute Q-values for next states
        next_q_values = self.target_model(next_states)

        # Compute target Q-values
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Create a mask for the actions taken
        mask = tf.one_hot(actions, self.action_size)

        # Compute the loss
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss(target_q_values, q_action)

        # Compute gradients and update the model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter += 1
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

        return loss
    
    def save(self, episode):
        if isinstance(episode, (int, float)):
            filename = f"models/{self.model_name}_{episode:.2f}.keras"
        else:
            filename = f"models/{self.model_name}_{episode}.keras"
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def load(self):
        return load_model(f"models/{self.model_name}.keras", custom_objects=self.custom_objects)