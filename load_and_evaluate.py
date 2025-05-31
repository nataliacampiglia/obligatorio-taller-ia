import torch
from utils import make_env, show_observation_stack
import gc

import importlib
import replay_memory  # or whatever module you changed
import dqn_cnn_model
import dqn_agent
import abstract_agent
importlib.reload(replay_memory)
importlib.reload(dqn_cnn_model)
importlib.reload(dqn_agent)
importlib.reload(abstract_agent)
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


def load_dqn_agent():
    torch.mps.empty_cache()
    gc.collect()
    env = make_env(ENV_NAME,
                video_folder='./videos/dqn_training',
                name_prefix="breakout",
                record_every=500,
                grayscale=GRAY_SCALE,
                screen_size=SCREEN_SIZE,
                stack_frames=NUM_STACKED_FRAMES,
                skip_frames=SKIP_FRAMES
                )
    net = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)

    # Cargar los pesos guardados
    net.load_state_dict(torch.load("GenericDQNAgent.dat", map_location=DEVICE))

    # Pasar a evaluación
    net.eval()

    # obtener DQN AGENT
    dqn_agent = DQNAgent(env, net, process_state, BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA, epsilon_i=EPSILON_INI, epsilon_f=EPSILON_MIN, epsilon_anneal_steps=EPSILON_ANNEAL_STEPS, episode_block=EPISODE_BLOCK, device=DEVICE)
    return dqn_agent