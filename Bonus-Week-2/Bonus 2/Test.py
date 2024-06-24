from statistics import median, mean
import gym
import numpy as np
import random
import tensorflow as tf
gym.envs.registration.register(
    id="CartPole-v1",
    entry_point="gym.envs.classic_control:CartPoleEnv",
    max_episode_steps=500, # Increase length of episode here. I have tried it upto 5000 steps. But it will take a long time.
    reward_threshold=475,
)
env = gym.make("CartPole-v1", render_mode="human")
env.reset()
model = tf.keras.models.load_model('Model_Perfected.h5')
scores = []
choices = []
for game in range(10): # Play 10 games. Win if you survive for 500 frames
    score = 0
    env.reset()
    prev_obs = []
    c = []
    done = False
    while not done:
        env.render()
        if(len(prev_obs)<=1):
            action = random.randint(0,1)
        else:
            action = np.argmax(model.predict(tf.convert_to_tensor([prev_obs]), verbose = 0, use_multiprocessing=True)) #argmax: Returns the indices of the maximum values along an axis.
        step_data = env.step(action)
        observation, reward, done, trunc, info = step_data
        prev_obs = observation
        score += reward
        c.append(action)
    print(score)
    scores.append(score)
    choices.append(c)
print(mean(scores))