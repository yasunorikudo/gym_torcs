import numpy as np
import matplotlib.pyplot as plt
import math

PI = math.pi

class Expert(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done, vision):
        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (3, 64, 64) <-- rgb
        # and values are in [0, 255]
        if vision is False:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, angle, trackPos = ob
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, angle, trackPos, vision = ob

            """ The code below is for checking the vision input. This is very heavy for real-time Control
                So you may need to remove.
            """

            img = np.ndarray((64,64,3))
            for i in range(3):
                img[:, :, i] = 255 - vision[i]

            plt.imshow(img, origin='lower')
            plt.draw()
            plt.pause(0.001)
            
        target_speed = 2

        accel = 0.5

        # Steer To Corner
        tmp = angle * 10 / PI
        # Steer To Center
        tmp -= trackPos * .10

        threshold = 0.1
        steering_value = 0.5

        if tmp > threshold:
            steer = steering_value
        elif tmp < - threshold:
            steer = - steering_value
        else:
            steer = 0

        # Throttle Control
        if speedX < target_speed - steer:
            brake = 0
        else:
            brake = 1
        # brake = np.random.randint(0, 2, 1)
        print (speedX)

        gear = None

        return steer, accel, brake, gear
