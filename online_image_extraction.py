from stable_baselines3 import PPO
import gym
import pybullet as p
import numpy as np
import math
import json
import cv2
import datetime
from environments.modulating_slope import modulatingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    version = "modulating_plainEnv/walk_straight"
    # load config from json
    with open(f"results/{version}/config.json") as f:
        config = json.load(f)

    # load trained vector environment
    gym_env = config['env']
    use_vae = config['use_vae']
    if use_vae:
        vec_env = DummyVecEnv([lambda: gym.make(gym_env, mode=0, freq_range=config['freq_range'], gamma=config['gamma'], vae_path=config['vae_path'])])
    else:
        vec_env = DummyVecEnv([lambda: gym.make(gym_env, mode=0, freq_range=config['freq_range'], gamma=config['gamma'])])
    vec_env = VecNormalize.load(f"results/{version}/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load(f"results/{version}/best_model.zip")

    # evaluate model
    obs = vec_env.reset()
    cpg_params = []
    motor_positions = []
    R = []
    poses = []
    vel = []
    for i in range(100000):

        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)

        # Take a picture every 10 steps:
        if i % 300 != 0:
            continue
        laikago = vec_env.envs[0].quadruped.laikago
        # position and orientation of the agent
        agent_pos, agent_orn = p.getBasePositionAndOrientation(laikago)
        euler = p.getEulerFromQuaternion(agent_orn)
        roll, pitch, yaw = euler
        # rotation matrices
        roll_rot = np.array(([1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]))
        pitch_rot = np.array(([math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]))
        yaw_rot = np.array(([math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]))
        unit_vec = np.array([0, 0, 1])
        camera_up = np.array([0, 1, 0])
        camera_dist = 1
        camera_targ = 10000
        rotated = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), unit_vec)
        rotated_point = camera_dist*rotated + agent_pos
        rotated_view = camera_targ*rotated + agent_pos
        rotated_up = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), camera_up)
        # camera view matrix
        view_matrix = p.computeViewMatrix(rotated_point, rotated_view, rotated_up)
        # camera projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.1, farVal=30.0)
        # get camera image
        image = p.getCameraImage(width=512, height=512, viewMatrix=view_matrix, projectionMatrix=projection_matrix, flags=p.ER_NO_SEGMENTATION_MASK)[2]    
    
        dt = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        cv2.imwrite(f"images/house_rgb_512/{dt}.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    