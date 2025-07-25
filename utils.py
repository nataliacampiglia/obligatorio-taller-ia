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

def collect_metrics(paths):
    all_rewards, all_losses, all_steps, all_epsilons = [], [], [], []
    episode_offset = 0
    for path in paths:
        data = np.load(path)
        rewards = data['rewards']
        losses = data['losses']
        steps = data['steps']
        epsilons = data['epsilons']
        num_episodes = len(rewards)
        episodes = np.arange(episode_offset, episode_offset + num_episodes)
        all_rewards.append((episodes, rewards))
        all_losses.append((episodes, losses))
        all_steps.append((episodes, steps))
        all_epsilons.append((episodes, epsilons))
        episode_offset += num_episodes
    return all_rewards, all_losses, all_steps, all_epsilons


def graph_metrics_accumulated(paths, phases=None, show_rewards=True, show_losses=True, show_steps=True, show_epsilons=True):
    """
    Genera gráficos acumulados de rewards, losses, steps y epsilons 
    a partir de múltiples archivos npz.

    Args:
        paths (list): Lista de rutas a archivos .npz.
        phases (list): Lista de nombres para cada fase (opcional).
    """
    plt.close('all')
    
    all_rewards, all_losses, all_steps, all_epsilons = collect_metrics(paths)

    if show_rewards:
        plt.figure(figsize=(10, 4))
        for idx, (episodes, rewards) in enumerate(all_rewards):
            label = phases[idx] if phases and idx < len(phases) else f"Fase {idx+1}"
            plt.plot(episodes, rewards, label=label)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()
    
    if show_epsilons:
        plt.figure(figsize=(10, 4))
        for idx, (episodes, epsilons) in enumerate(all_epsilons):
            label = phases[idx] if phases and idx < len(phases) else f"Fase {idx+1}"
            plt.plot(episodes, epsilons, label=label)
        plt.title("Epsilon por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    if show_losses:
        plt.figure(figsize=(10, 4))
        for idx, (episodes, losses) in enumerate(all_losses):
            label = phases[idx] if phases and idx < len(phases) else f"Fase {idx+1}"
            plt.plot(episodes, losses, label=label)
        plt.title("Loss por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    if show_steps:
        plt.figure(figsize=(10, 4))
        for idx, (episodes, steps) in enumerate(all_steps):
            label = phases[idx] if phases and idx < len(phases) else f"Fase {idx+1}"
            plt.plot(episodes, steps, label=label)
        plt.title("Steps por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()


def graph_metrics_comparison(dqn_path, ddqn_path, show_rewards=True, show_losses=True, show_steps=True, show_epsilons=True):
    """
    Grafica comparativamente las métricas de entrenamiento (rewards, losses, steps, epsilons)
    para DQN y DDQN en la misma figura.

    Args:
        dqn_path (list): Lista de rutas a archivos .npz de DQN.
        ddqn_path (list): Lista de rutas a archivos .npz de DDQN.
    """
    plt.close('all')

    dqn_rewards, dqn_losses, dqn_steps, dqn_epsilons = collect_metrics(dqn_path)
    ddqn_rewards, ddqn_losses, ddqn_steps, ddqn_epsilons = collect_metrics(ddqn_path)

    # Recompensa
    if show_rewards:
        plt.figure(figsize=(12, 5))
        # Concatenar todas las fases de DQN
        all_dqn_episodes = []
        all_dqn_rewards = []
        for episodes, rewards in dqn_rewards:
            all_dqn_episodes.extend(episodes)
            all_dqn_rewards.extend(rewards)
        plt.plot(all_dqn_episodes, all_dqn_rewards, label='DQN', color='tab:blue', linestyle='-')
        
        # Concatenar todas las fases de DDQN
        all_ddqn_episodes = []
        all_ddqn_rewards = []
        for episodes, rewards in ddqn_rewards:
            all_ddqn_episodes.extend(episodes)
            all_ddqn_rewards.extend(rewards)
        plt.plot(all_ddqn_episodes, all_ddqn_rewards, label='DDQN', color='tab:orange', linestyle='--')
        
        plt.title("Recompensa por Episodio (DQN vs DDQN)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    # Epsilon
    if show_epsilons:
        plt.figure(figsize=(12, 5))
        # Concatenar todas las fases de DQN
        all_dqn_episodes = []
        all_dqn_epsilons = []
        for episodes, epsilons in dqn_epsilons:
            all_dqn_episodes.extend(episodes)
            all_dqn_epsilons.extend(epsilons)
        plt.plot(all_dqn_episodes, all_dqn_epsilons, label='DQN', color='tab:blue', linestyle='-')
        
        # Concatenar todas las fases de DDQN
        all_ddqn_episodes = []
        all_ddqn_epsilons = []
        for episodes, epsilons in ddqn_epsilons:
            all_ddqn_episodes.extend(episodes)
            all_ddqn_epsilons.extend(epsilons)
        plt.plot(all_ddqn_episodes, all_ddqn_epsilons, label='DDQN', color='tab:orange', linestyle='--')
        
        plt.title("Epsilon por Episodio (DQN vs DDQN)")
        plt.xlabel("Episodio")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    # Loss
    if show_losses:
        plt.figure(figsize=(12, 5))
        # Concatenar todas las fases de DQN
        all_dqn_episodes = []
        all_dqn_losses = []
        for episodes, losses in dqn_losses:
            all_dqn_episodes.extend(episodes)
            all_dqn_losses.extend(losses)
        plt.plot(all_dqn_episodes, all_dqn_losses, label='DQN', color='tab:blue', linestyle='-')
        
        # Concatenar todas las fases de DDQN
        all_ddqn_episodes = []
        all_ddqn_losses = []
        for episodes, losses in ddqn_losses:
            all_ddqn_episodes.extend(episodes)
            all_ddqn_losses.extend(losses)
        plt.plot(all_ddqn_episodes, all_ddqn_losses, label='DDQN', color='tab:orange', linestyle='--')
        
        plt.title("Loss por Episodio (DQN vs DDQN)")
        plt.xlabel("Episodio")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    # Steps
    if show_steps:
        plt.figure(figsize=(12, 5))
        # Concatenar todas las fases de DQN
        all_dqn_episodes = []
        all_dqn_steps = []
        for episodes, steps in dqn_steps:
            all_dqn_episodes.extend(episodes)
            all_dqn_steps.extend(steps)
        plt.plot(all_dqn_episodes, all_dqn_steps, label='DQN', color='tab:blue', linestyle='-')
        
        # Concatenar todas las fases de DDQN
        all_ddqn_episodes = []
        all_ddqn_steps = []
        for episodes, steps in ddqn_steps:
            all_ddqn_episodes.extend(episodes)
            all_ddqn_steps.extend(steps)
        plt.plot(all_ddqn_episodes, all_ddqn_steps, label='DDQN', color='tab:orange', linestyle='--')
        
        plt.title("Steps por Episodio (DQN vs DDQN)")
        plt.xlabel("Episodio")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    
def compare_metrics(metrics_paths, labels=None, show_rewards=True, show_losses=True, show_steps=False, show_actions=False, show_epsilons=True):
    """
    Compara métricas de entrenamiento entre diferentes fases o agentes.
    
    Args:
        metrics_paths (list): Lista de rutas a los archivos .npz con las métricas
        labels (list): Lista de etiquetas para cada conjunto de métricas. Si es None, usa los nombres de archivo
        show_rewards (bool): Si mostrar comparación de recompensas
        show_losses (bool): Si mostrar comparación de losses
        show_steps (bool): Si mostrar comparación de steps
        show_actions (bool): Si mostrar comparación de acciones
        show_epsilons (bool): Si mostrar comparación de epsilons
    """
    if labels is None:
        labels = [os.path.basename(path).replace('.npz', '') for path in metrics_paths]
    
    # Cargar todos los datos
    all_data = []
    for path in metrics_paths:
        data = np.load(path)
        all_data.append({
            'rewards': data['rewards'],
            'losses': data['losses'],
            'steps': data['steps'],
            'epsilons': data['epsilons'],
            'actions': data['actions']
        })
    
    # Recompensas
    if show_rewards:
        plt.figure(figsize=(12, 6))
        for i, (data, label) in enumerate(zip(all_data, labels)):
            plt.plot(data['rewards'], label=label, alpha=0.8)
        plt.title("Comparación de Recompensas por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Epsilon
    if show_epsilons:
        plt.figure(figsize=(12, 6))
        for i, (data, label) in enumerate(zip(all_data, labels)):
            plt.plot(data['epsilons'], label=label, alpha=0.8)
        plt.title("Comparación de Epsilon por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Loss
    if show_losses:
        plt.figure(figsize=(12, 6))
        for i, (data, label) in enumerate(zip(all_data, labels)):
            plt.plot(data['losses'], label=label, alpha=0.8)
        plt.title("Comparación de Loss por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Steps
    if show_steps:
        plt.figure(figsize=(12, 6))
        for i, (data, label) in enumerate(zip(all_data, labels)):
            plt.plot(data['steps'], label=label, alpha=0.8)
        plt.title("Comparación de Steps por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Steps")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Actions (promedio de las últimas N épocas para estabilidad)
    if show_actions:
        n_episodes = 100  # Últimas 100 épocas para estabilidad
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for action_idx in range(4):
            ax = axes[action_idx]
            for i, (data, label) in enumerate(zip(all_data, labels)):
                # Tomar promedio de las últimas n_episodes
                recent_actions = data['actions'][-n_episodes:, action_idx]
                avg_action = np.mean(recent_actions)
                ax.bar(label, avg_action, alpha=0.8, label=label)
            
            ax.set_title(f"Promedio {action_names[action_idx]} (últimas {n_episodes} épocas)")
            ax.set_ylabel("Frecuencia promedio")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Resumen estadístico
    print("\n=== RESUMEN ESTADÍSTICO ===")
    for i, (data, label) in enumerate(zip(all_data, labels)):
        print(f"\n{label}:")
        
        # Limpiar datos de recompensas (remover nan e inf)
        rewards_clean = data['rewards'][~np.isnan(data['rewards']) & ~np.isinf(data['rewards'])]
        if len(rewards_clean) > 0:
            print(f"  Recompensa promedio: {np.mean(rewards_clean):.2f} ± {np.std(rewards_clean):.2f}")
            print(f"  Recompensa máxima: {np.max(rewards_clean):.2f}")
            print(f"  Recompensa mínima: {np.min(rewards_clean):.2f}")
        else:
            print(f"  Recompensa promedio: NaN (no hay datos válidos)")
            print(f"  Recompensa máxima: NaN (no hay datos válidos)")
        
        # Limpiar datos de losses
        losses_clean = data['losses'][~np.isnan(data['losses']) & ~np.isinf(data['losses'])]
        if len(losses_clean) > 0:
            print(f"  Loss promedio: {np.mean(losses_clean):.4f} ± {np.std(losses_clean):.4f}")
        else:
            print(f"  Loss promedio: NaN (no hay datos válidos)")
        
        # Limpiar datos de steps
        steps_clean = data['steps'][~np.isnan(data['steps']) & ~np.isinf(data['steps'])]
        if len(steps_clean) > 0:
            print(f"  Steps promedio: {np.mean(steps_clean):.1f} ± {np.std(steps_clean):.1f}")
        else:
            print(f"  Steps promedio: NaN (no hay datos válidos)")
        
        # Epsilon final
        if not np.isnan(data['epsilons'][-1]) and not np.isinf(data['epsilons'][-1]):
            print(f"  Epsilon final: {data['epsilons'][-1]:.4f}")
        else:
            print(f"  Epsilon final: NaN")
        
        # Información adicional sobre datos
        total_episodes = len(data['rewards'])
        valid_rewards = len(rewards_clean)
        print(f"  Episodios totales: {total_episodes}")
        print(f"  Episodios con recompensas válidas: {valid_rewards}")
        if total_episodes > 0:
            print(f"  Porcentaje de datos válidos: {(valid_rewards/total_episodes)*100:.1f}%")