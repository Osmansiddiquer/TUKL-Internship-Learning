from statistics import median, mean
import gym
import numpy as np
import random
import tensorflow as tf
import math
env = gym.make("CartPole-v1")
env.reset()
goal_steps  = 500
min_score = 75
initial_games = 30000
LR = 1e-3
def random_episodes():
    for episodes in range(5):
        for i in range(goal_steps):
            action  = env.action_space.sample()
            observation, reward, done, trunc, info = env.step(action)
            print(action)
            if done:
                env.reset()
                break

def train_model(training_data, model):
    X = tf.convert_to_tensor(np.array([i[0] for i in training_data]))
    y= tf.convert_to_tensor(np.array([np.array(i[1]) for i in training_data]))
    model.fit(X, y, epochs = 10)
    return model

def neural_network_model(input_size, LR):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

#random_episodes()
def gathering_train_data():
    training_data = []
    scores = []
    accepted_scores  = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for i in range(goal_steps):
            action  = random.randint(0,1)
            observation, reward, done, info = env.step(action)
            if(len(previous_observation)>1):
                game_memory.append([previous_observation, action])
                score += 0.1*(math.fabs(previous_observation[2]) - math.fabs(observation[2]))
                score += 0.1*(math.fabs(previous_observation[0]) - math.fabs(observation[0]))
                score -= 0.01*(math.exp(math.fabs(observation[2]))+math.exp(math.fabs(observation[1])))
            previous_observation = observation
            if done:
                break
            score += reward
        scores.append(score) #keeping track
        if(score>=min_score):
            accepted_scores.append(score)
            #one-hot encoding. Necessary for non-binary actions
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                else:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
    np.save("x_train.npy", np.array(training_data))
    print(mean(accepted_scores))
    print(median(accepted_scores))
    print(accepted_scores)
    print(len(accepted_scores))
    print(len(scores))
    return training_data, accepted_scores
training_data, scores = gathering_train_data()


model = neural_network_model(4, LR)

model = train_model(training_data, model)
model.save("Model1.h5")
