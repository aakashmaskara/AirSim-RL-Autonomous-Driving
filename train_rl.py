from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from airsim_rl_env import AirSimCarEnv

# Function to create environment
def make_env():
    return AirSimCarEnv()

# Create vectorized environment
env = make_vec_env(make_env, n_envs=1)

# Load PPO model
model = PPO("MultiInputPolicy", env, learning_rate=0.001, verbose=1, tensorboard_log="./ppo_tensorboard/")

# Start Training
print("🚀 Starting training with Reinforcement Learning...")
model.learn(total_timesteps=200000)

# Save Model
model.save("airsim_rl_model")
print("✅ Model training complete & saved!")
