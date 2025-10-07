import airsim
import numpy as np
import gym
from gym import spaces
import time

class AirSimCarEnv(gym.Env):
    def __init__(self):
        super(AirSimCarEnv, self).__init__()

        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.iteration = 0

        # Observation space: Depth image + Velocity
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.float32),
            "velocity": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
        })

        # Action space: Steering, Throttle, Brake
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

    def seed(self, seed=None):  # ✅ FIXED! Added seed function
        np.random.seed(seed)

    def reset(self):
        self.iteration += 1
        print(f"🚗 Respawning car... (Iteration {self.iteration})")
        
        self.client.reset()
        time.sleep(1)
        self.client.enableApiControl(True)
        self.car_controls.throttle = 0
        self.car_controls.brake = 0
        self.client.setCarControls(self.car_controls)

        return {"depth": self._get_depth_image(), "velocity": np.array([self._get_velocity()], dtype=np.float32)}

    def step(self, action):
        self.car_controls.steering = float(action[0])  # Left/Right
        self.car_controls.throttle = float(action[1])  # Forward
        self.car_controls.brake = float(action[2])  # Stop
        self.client.setCarControls(self.car_controls)

        # Get observations
        depth_img = self._get_depth_image()
        velocity = np.array([self._get_velocity()], dtype=np.float32)

        # Check collision
        collision_info = self.client.simGetCollisionInfo()
        done = collision_info.has_collided
        reward = -10 if done else 1  # Heavy penalty for collision

        if done:
            print(f"❌ Collision at {collision_info.impact_point} - Respawning immediately!")

        return {"depth": depth_img, "velocity": velocity}, reward, done, {}

    def _get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        img = np.array(responses[0].image_data_float, dtype=np.float32)
        img = img.reshape((144, 256))  # Original shape
        img = img[30:114, 86:170]  # Crop to (84x84)
        img = img / 255.0  # Normalize
        return img.reshape((84, 84, 1))

    def _get_velocity(self):
        car_state = self.client.getCarState()
        return car_state.speed

    def close(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
