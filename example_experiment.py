from gym_torcs import TorcsEnv
from expert_agent import Expert
import numpy as np

vision = True
throttle = True
gear_change = False
episode_count = 10
max_steps = 50000
reward = 0
done = False
step = 0

# Generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=throttle, gear_change=gear_change)

expert = Expert(4)  # steer, accel, brake and gear


print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    total_reward = 0.
    for j in range(max_steps):
        action = expert.act(ob, reward, done, vision)
        steer, accel, brake, gear = action

        ob, reward, done, _ = env.step(action)
        #print(ob)
        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

env.end()  # This is for shutting down TORCS
print("Finish.")
