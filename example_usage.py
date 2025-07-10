"""
Ejemplo de uso de memoria de repetición regular vs priorizada
"""

import torch
import gymnasium as gym
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dqn_cnn_model import DQNCNN
from replay_memory import ReplayMemoryFactory

def example_regular_replay():
    """
    Ejemplo usando memoria de repetición regular
    """
    print("=== Ejemplo con memoria de repetición regular ===")
    
    # Configuración del entorno
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear modelo
    model = DQNCNN(env.action_space.n)
    
    # Función de procesamiento de observaciones (ejemplo simple)
    def process_obs(obs):
        return torch.from_numpy(obs).float().to(device)
    
    # Crear agente con memoria regular (por defecto)
    agent = DQNAgent(
        env=env,
        model=model,
        obs_processing_func=process_obs,
        memory_buffer_size=10000,
        batch_size=32,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_i=1.0,
        epsilon_f=0.01,
        epsilon_anneal_steps=1000000,
        episode_block=100,
        device=device,
        run_name="regular_replay_example",
        use_prioritized_replay=False  # Memoria regular
    )
    
    print(f"Tipo de memoria: {type(agent.memory).__name__}")
    print(f"Usando memoria priorizada: {agent.use_prioritized_replay}")
    
    # Entrenar por pocos episodios para demostración
    print("Entrenando con memoria regular...")
    agent.train(number_episodes=5, max_steps=1000)
    
    env.close()

def example_prioritized_replay():
    """
    Ejemplo usando memoria de repetición priorizada
    """
    print("\n=== Ejemplo con memoria de repetición priorizada ===")
    
    # Configuración del entorno
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear modelo
    model = DQNCNN(env.action_space.n)
    
    # Función de procesamiento de observaciones (ejemplo simple)
    def process_obs(obs):
        return torch.from_numpy(obs).float().to(device)
    
    # Crear agente con memoria priorizada
    agent = DQNAgent(
        env=env,
        model=model,
        obs_processing_func=process_obs,
        memory_buffer_size=10000,
        batch_size=32,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_i=1.0,
        epsilon_f=0.01,
        epsilon_anneal_steps=1000000,
        episode_block=100,
        device=device,
        run_name="prioritized_replay_example",
        use_prioritized_replay=True,  # Memoria priorizada
        prioritized_replay_alpha=0.6,  # Parámetro de priorización
        prioritized_replay_beta=0.4,   # Parámetro de importancia sampling
        prioritized_replay_beta_increment=0.001,  # Incremento de beta
        prioritized_replay_epsilon=1e-6  # Valor pequeño para evitar prioridades cero
    )
    
    print(f"Tipo de memoria: {type(agent.memory).__name__}")
    print(f"Usando memoria priorizada: {agent.use_prioritized_replay}")
    print(f"Alpha: {agent.memory.alpha}")
    print(f"Beta: {agent.memory.beta}")
    
    # Entrenar por pocos episodios para demostración
    print("Entrenando con memoria priorizada...")
    agent.train(number_episodes=5, max_steps=1000)
    
    env.close()

def example_factory_usage():
    """
    Ejemplo usando el factory para crear diferentes tipos de memoria
    """
    print("\n=== Ejemplo usando ReplayMemoryFactory ===")
    
    # Crear memoria regular usando factory
    regular_memory = ReplayMemoryFactory.create_memory(
        memory_type="regular",
        capacity=1000,
        device=torch.device("cpu")
    )
    print(f"Memoria regular creada: {type(regular_memory).__name__}")
    
    # Crear memoria priorizada usando factory
    prioritized_memory = ReplayMemoryFactory.create_memory(
        memory_type="prioritized",
        capacity=1000,
        device=torch.device("cpu"),
        alpha=0.6,
        beta=0.4
    )
    print(f"Memoria priorizada creada: {type(prioritized_memory).__name__}")
    
    # Agregar algunas transiciones de ejemplo
    dummy_state = torch.randn(4, 84, 84)
    dummy_next_state = torch.randn(4, 84, 84)
    
    regular_memory.add(dummy_state, 0, 1.0, False, dummy_next_state)
    prioritized_memory.add(dummy_state, 0, 1.0, False, dummy_next_state)
    
    print(f"Transiciones en memoria regular: {len(regular_memory)}")
    print(f"Transiciones en memoria priorizada: {len(prioritized_memory)}")

def example_double_dqn_prioritized():
    """
    Ejemplo usando Double DQN con memoria priorizada
    """
    print("\n=== Ejemplo Double DQN con memoria priorizada ===")
    
    # Configuración del entorno
    env = gym.make('ALE/Breakout-v5', render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear modelos para Double DQN
    model_a = DQNCNN(env.action_space.n)
    model_b = DQNCNN(env.action_space.n)
    
    # Función de procesamiento de observaciones
    def process_obs(obs):
        return torch.from_numpy(obs).float().to(device)
    
    # Crear agente Double DQN con memoria priorizada
    agent = DoubleDQNAgent(
        gym_env=env,
        model_a=model_a,
        model_b=model_b,
        obs_processing_func=process_obs,
        memory_buffer_size=10000,
        batch_size=32,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_i=1.0,
        epsilon_f=0.01,
        epsilon_anneal_steps=1000000,
        episode_block=100,
        device=device,
        sync_target=1000,
        run_name="double_dqn_prioritized_example",
        use_clip=True,
        use_prioritized_replay=True,  # Memoria priorizada
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        prioritized_replay_beta_increment=0.001,
        prioritized_replay_epsilon=1e-6
    )
    
    print(f"Tipo de memoria: {type(agent.memory).__name__}")
    print(f"Usando memoria priorizada: {agent.use_prioritized_replay}")
    
    # Entrenar por pocos episodios para demostración
    print("Entrenando Double DQN con memoria priorizada...")
    agent.train(number_episodes=5, max_steps=1000)
    
    env.close()

if __name__ == "__main__":
    print("Ejemplos de uso de memoria de repetición regular vs priorizada")
    print("=" * 60)
    
    # Ejecutar ejemplos
    example_regular_replay()
    example_prioritized_replay()
    example_factory_usage()
    example_double_dqn_prioritized()
    
    print("\n" + "=" * 60)
    print("Todos los ejemplos completados exitosamente!") 