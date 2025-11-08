#!/usr/bin/env python3
"""
Train a DQN agent to play Atari Breakout using Keras-RL2 and Gymnasium.
Saves the final trained policy network as 'policy.h5'.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def build_model(input_shape, nb_actions):
    """Build a convolutional neural network for the DQN agent."""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def main():
    # Create and wrap the Breakout environment
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)

    nb_actions = env.action_space.n
    input_shape = (4, 84, 84)  # FrameStack + AtariPreprocessing output shape

    # Build the model
    model = build_model(input_shape, nb_actions)
    print(model.summary())

    # Configure memory and exploration policy
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    # Create and compile the DQN agent
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the agent — you can adjust nb_steps for longer training
    print("Training started...")
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)
    print("Training finished!")

    # Save the trained weights
    dqn.save_weights('policy.h5', overwrite=True)
    print("Saved trained policy as 'policy.h5'")

    env.close()


if __name__ == '__main__':
    main()
