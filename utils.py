import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import (
    TransformReward,
    RecordVideo,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    AtariPreprocessing,
    FrameStackObservation
)
from constants import (DQN_TYPE)

def show_observation(observation):
    dimension = observation.shape
    if len(dimension) == 3:
        if dimension[2] == 3:
            plt.imshow(observation)
        elif dimension[2] == 1:
            plt.imshow(observation[:, :, 0], cmap='gray')
    elif len(dimension) == 2:
        plt.imshow(observation, cmap='gray')
    else:
        raise ValueError("Invalid observation shape")
    plt.show()
    
def show_observation_stack(observation):
    frames = observation.shape[0]
    for i in range(frames):
        show_observation(observation[i])


class FireOnLifeLostWrapper(gymnasium.Wrapper):
    """Presiona FIRE automáticamente tras reset y tras cada pérdida de vida."""
    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def reset(self, **kwargs):
        # 1) Reset normal
        obs, info = self.env.reset(**kwargs)
        # 2) Inyectar FIRE para arrancar la partida
        obs, _, terminated, truncated, info = self.env.step(1)
        # Si por alguna razón el juego acabó (raro), reinicia otra vez
        if terminated or truncated:
            return self.reset(**kwargs)
        # 3) Guarda el número de vidas inicial
        self._prev_lives = info.get('lives')
        return obs, info

    def step(self, action):
        # 1) Paso normal del agente
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 2) Detecta pérdida de vida
        current_lives = info.get('lives', self._prev_lives)
        if (current_lives < self._prev_lives) and not (terminated or truncated):
            # 3) Inyecta FIRE para reanudar tras perder vida
            obs, fire_reward, terminated, truncated, info = self.env.step(1)
            reward += fire_reward  # opcional: sumar recompensa de FIRE
        # 4) Actualiza contador de vidas
        self._prev_lives = current_lives
        return obs, reward, terminated, truncated, info

def make_env(
    env_name: str,
    render_mode: str = "rgb_array",
    # Video
    video_folder: str | None = "./videos",
    name_prefix: str = "",
    record_every: int | None = None,
    # Preprocesado
    grayscale: bool = False,
    screen_size: int = 84,
    stack_frames: int = 4,
    skip_frames: int = 4
) -> gymnasium.Env:

    env = gymnasium.make(env_name, render_mode=render_mode, frameskip=1)
    
    if video_folder is not None and record_every is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep % record_every == 0,
            fps=env.metadata.get("render_fps", 30) * skip_frames,
        )
    
    # env = FireOnLifeLostWrapper(env)
    
    env = AtariPreprocessing(
        env,
        noop_max=10,
        frame_skip=skip_frames,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=False
    )
    
    # stack frames
    env = FrameStackObservation(env, stack_size=stack_frames)
    
    # clip rewards
    sign_fn = lambda r: 1 if r > 0 else (-1 if r < 0 else 0)
    env = TransformReward(env, sign_fn)
    
    return env

def evaluate_training_phase_results(pathname="", phase_id="", type=DQN_TYPE): 
    graph_metrics(pathname)
    plot_q_values_per_phase(phase_id, type)

def graph_metrics(pathname="", show_rewards=True, show_losses=True, show_steps=True, show_actions=True, show_epsilons=True):
    """
    Grafica las métricas de entrenamiento.
    Args:
        metrics (dict): Diccionario con las métricas a graficar.
        title (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.
        filename (str): Nombre del archivo para guardar el gráfico. Si está vacío, no se guarda.
    """
    data = np.load(pathname)

    rewards = data["rewards"]
    losses = data["losses"]
    steps = data["steps"]
    epsilons = data["epsilons"]
    actions = data["actions"]
    print(f"Datos cargados de {pathname}:",  data["actions"])

    # Recompensas
    if show_rewards:
        plt.figure(figsize=(10, 4))
        plt.plot(rewards)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.grid(True)
        plt.show()
    
    # Epsilon
    if show_epsilons:
        plt.figure(figsize=(10, 4))
        plt.plot(epsilons)
        plt.title("Epsilon por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.show()

    # Loss
    if show_losses:
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.title("Loss por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    # Steps
    if show_steps:
        plt.figure(figsize=(10, 4))
        plt.plot(steps)
        plt.title("Steps por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.show()

    # Actions
    if show_actions:
        plt.figure(figsize=(10, 4))
        for i in range(actions.shape[1]):
            plt.plot(actions[:, i], label=f"Acción {i == 0 and 'NOOP' or i == 1 and 'FIRE' or i == 2 and 'RIGHT' or 'LEFT'}")
        plt.title("Distribución de Acciones por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(True)
        plt.show() 
    
def load_q_values(filename, type='dqn'):
    path = os.path.join(f"q_values/{type}", f"{filename}.npz")
    data = np.load(path)
    return data['q_values']

def plot_q_values_per_phase(phase_id, type):
    q_values = load_q_values(phase_id, type)
    max_q_per_state = np.max(q_values, axis=1)
    
    plt.figure(figsize=(10,6))
    plt.plot(max_q_per_state)
    plt.title(f"Curva de convergencia Q - {phase_id}")
    plt.xlabel("Estado de referencia")
    plt.ylabel("Q máximo")
    plt.grid(True)
    plt.show()

def compare_q_values_across_phases(phase_names):
    plt.figure(figsize=(12, 7))
    
    for phase in phase_names:
        q_values = load_q_values(phase)
        max_q_per_state = np.max(q_values, axis=1)
        plt.plot(max_q_per_state, label=phase)
    
    plt.title("Comparativa de convergencia Q por fase")
    plt.xlabel("Estado de referencia")
    plt.ylabel("Q máximo")
    plt.grid(True)
    plt.legend()
    plt.show()

def q_values_summary(phase_names):
    summary = {}
    for phase in phase_names:
        q_values = load_q_values(phase)
        max_q_per_state = np.max(q_values, axis=1)
        summary[phase] = {
            'mean': np.mean(max_q_per_state),
            'std': np.std(max_q_per_state),
            'min': np.min(max_q_per_state),
            'max': np.max(max_q_per_state),
        }
    return summary