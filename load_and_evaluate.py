import torch
from utils import make_env, show_observation_stack
import gc

import importlib
import replay_memory  # or whatever module you changed
import dqn_cnn_model
import dqn_agent
import abstract_agent
import constants
importlib.reload(replay_memory)
importlib.reload(dqn_cnn_model)
importlib.reload(dqn_agent)
importlib.reload(abstract_agent)
importlib.reload(constants)
from dqn_cnn_model import DQN_CNN_Model
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
import numpy as np
import os
from IPython.display import Video

from constants import (
    ENV_NAME, DEVICE, GRAY_SCALE, SCREEN_SIZE, NUM_STACKED_FRAMES, SKIP_FRAMES,
    BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA, EPSILON_INI, EPSILON_MIN, EPSILON_ANNEAL_STEPS, EPISODE_BLOCK, TOTAL_STEPS, EPISODES, STEPS_PER_EPISODE, DQN_TYPE
)

def process_state(obs):
    """
    Preprocess the state to be used as input for the model (transform to tensor).
    """
    return torch.tensor(obs, dtype=torch.float32, device=DEVICE) / 255.0


def create_env(video_folder='./videos/dqn_training'):
    env = make_env(ENV_NAME,
        video_folder=video_folder,
        name_prefix="breakout",
        record_every=500,
        grayscale=GRAY_SCALE,
        screen_size=SCREEN_SIZE,
        stack_frames=NUM_STACKED_FRAMES,
        skip_frames=SKIP_FRAMES
        )
    return env



def load_dqn_agent(env,loadPath=None, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, gamma=GAMMA,
                    epsilon_i=EPSILON_INI, epsilon_f=EPSILON_MIN,epsilon_anneal_steps= EPSILON_ANNEAL_STEPS, episode_block=EPISODE_BLOCK, run_name="run", use_prioritized_replay=False):
    ### Creamos la policy_net
    net = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)

    # Cargar los pesos guardados
    if (loadPath):
        net.load_state_dict(torch.load(loadPath, map_location=DEVICE))

    # Pasar a evaluación
    net.eval()

    print("Parametros del agente:")
    print(f"loadPath: {loadPath}")
    print(f"buffer_size: {buffer_size}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"gamma: {gamma}")
    print(f"epsilon_i: {epsilon_i}")
    print(f"epsilon_f: {epsilon_f}")
    print(f"epsilon_anneal_steps: {epsilon_anneal_steps}")
    print(f"episode_block: {episode_block}")
    print(f"run_name: {run_name}")
    print(f"Use prioritized memory: {use_prioritized_replay}")

  # obtener DQN AGENT
    dqn_agent = DQNAgent(env, net, process_state, buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f,epsilon_anneal_steps, episode_block, device=DEVICE, run_name=run_name, use_prioritized_replay=use_prioritized_replay)
    return dqn_agent

def create_reference_states():
    env = create_env(video_folder='./videos/reference_states')
    reference_states = []
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            state, _, done, _, _ = env.step(action)
            if done:
                break
        state_phi = process_state(state)
        reference_states.append(state_phi)
    env.close()
    return reference_states

def save_q_values(q_values_dir, policy_net, reference_states, device, filename):
    policy_net.eval()
    with torch.no_grad():
        q_values = []
        for state in reference_states:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q = policy_net(state_tensor).cpu().numpy().squeeze()
            q_values.append(q)
        q_values = np.array(q_values)
        os.makedirs(q_values_dir, exist_ok=True)
        savePath = os.path.join(q_values_dir, f"{filename}.npz")
        np.savez(savePath, q_values=q_values)


def execute_dqn_training_phase(phase_id, reference_states, loadPath = None, total_steps = TOTAL_STEPS, episodes = EPISODES, epsilon_i = EPSILON_INI, epsilon_f = EPSILON_MIN, epsilon_anneal_steps = EPSILON_ANNEAL_STEPS, gamma=GAMMA, use_prioritized_replay=False):
    video_folder = f'./videos/dqn/{phase_id}'
    env = create_env(video_folder=video_folder)
    dqn_agent = load_dqn_agent(env, loadPath=loadPath, epsilon_i=epsilon_i, epsilon_f=epsilon_f, epsilon_anneal_steps=epsilon_anneal_steps, episode_block=EPISODE_BLOCK, run_name=phase_id, gamma=gamma, use_prioritized_replay=use_prioritized_replay)
    dqn_agent.train(episodes, STEPS_PER_EPISODE, total_steps)
    save_q_values("q_values/dqn", dqn_agent.policy_net, reference_states, DEVICE, f"{phase_id}")
    env.close()
    return dqn_agent

def execute_ddqn_training_phase(phase_id, reference_states, load_net_path = None, total_steps = TOTAL_STEPS, episodes = EPISODES, epsilon_ini = EPSILON_INI, epsilon_min = EPSILON_MIN, epsilon_anneal_steps = EPSILON_ANNEAL_STEPS, gamma=GAMMA):
    print("Parametros del agente:")
    print(f"loadPath: {load_net_path}")
    print(f"gamma: {gamma}")
    print(f"epsilon_i: {epsilon_ini}")
    print(f"epsilon_f: {epsilon_min}")
    print(f"epsilon_anneal_steps: {epsilon_anneal_steps}")
    print(f"run_name: {phase_id}")
    video_folder = f'./videos/ddqn/{phase_id}'
    env = create_env(video_folder=video_folder)
    # Capturar los estados referentes p-ara visualizar la convergencia de los valores de Q
    modelo_a = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)
    if(load_net_path is not None):
        modelo_a.load_state_dict(torch.load(load_net_path))
    modelo_b = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)
    ddqn_agent = DoubleDQNAgent(env, modelo_a, modelo_b, process_state, BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, gamma=gamma, epsilon_i= epsilon_ini, epsilon_f=epsilon_min, epsilon_anneal_steps=epsilon_anneal_steps, episode_block = EPISODE_BLOCK, device=DEVICE, run_name=phase_id)
    ddqn_agent.train(episodes, STEPS_PER_EPISODE, total_steps)
    save_q_values("q_values/ddqn", ddqn_agent.online_net, reference_states, DEVICE, f"{phase_id}")
    env.close()
    return ddqn_agent

def execute_agent_play(agent, phase_id, type=DQN_TYPE):
    VALIDATION_VIDEO_FOLDER = f'./videos/{type}/validation/{phase_id}'
    # create env
    env = create_env(video_folder=VALIDATION_VIDEO_FOLDER)
    # play
    agent.play(env, episodes=1)

    env.close()

    # Ruta al archivo de vídeo en tu sistema de ficheros
    video_path = f"{VALIDATION_VIDEO_FOLDER}/breakout-episode-0.mp4"
    return video_path
   


