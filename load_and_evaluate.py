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

from constants import (
    ENV_NAME, DEVICE, GRAY_SCALE, SCREEN_SIZE, NUM_STACKED_FRAMES, SKIP_FRAMES,
    BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA, EPSILON_INI, EPSILON_MIN, EPSILON_ANNEAL_STEPS, EPISODE_BLOCK
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
                    epsilon_i=EPSILON_INI, epsilon_f=EPSILON_MIN,epsilon_anneal_steps= EPSILON_ANNEAL_STEPS, episode_block=EPISODE_BLOCK, run_name="run"):
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

  # obtener DQN AGENT
    dqn_agent = DQNAgent(env, net, process_state, buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f,epsilon_anneal_steps, episode_block, device=DEVICE, run_name=run_name)
    return dqn_agent
