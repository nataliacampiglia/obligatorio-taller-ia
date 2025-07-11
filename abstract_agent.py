import os
import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
from datetime import datetime

from constants import (DDQN_NET_HISTORY_DIR, DQN_BREAKPOINT_DIR, DDQN_BREAKPOINT_DIR, DQN_COMMON_MODEL_PATH, EPSILON_ADAPTIVE_PATIENCE, EPSILON_ADAPTIVE_INCREASE, EPSILON_ADAPTIVE_DECREASE,
                    getMetricsDir, getMetricFilePath, getGenericDataFilePath
)

class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device,  run_name="run", checkpoint_every=150000, load_checkpoint=None, adaptive_epsilon=False):
        self.device = device

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente 
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma


        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        
        self.episode_block = episode_block
        self.checkpoint_every = checkpoint_every
        self.load_checkpoint = load_checkpoint
        self.run_name = run_name

        self.total_steps = 0
        self.all_actions = []

        # Variables de control para la implementación de epsilon adaptativo
        self.adaptive_epsilon = adaptive_epsilon
        self.best_reward = -float('inf')
        self.no_improvement_episodes = 0
        self.patience = EPSILON_ADAPTIVE_PATIENCE # Numero de episodios sin mejorar antes de aumentar epsilon
        self.epsilon_increase = EPSILON_ADAPTIVE_INCREASE
        self.epsilon_decrease = EPSILON_ADAPTIVE_DECREASE
        self.epsilon_min = self.epsilon_f
        self.epsilon_max = self.epsilon_i
    
    def train(self, number_episodes = 50_000, max_steps_episode = 10_000, max_steps=1_000_000):
      rewards = []
      mean_rewards = []
      losses = []
      epsilons = []
      steps_per_episode = []
      total_steps = 0
      
      metrics = {"reward": 0.0, "epsilon": self.epsilon_i, "steps": 0}

      pbar = tqdm(range(number_episodes), desc="Entrenando", unit="episode")
      print('\n===========================================================================================\n')
      print(f"Iniciando entrenamiento {self.run_name}, con los siguientes hiperparametros:\n")
      print(f"gamma: {self.gamma}, epsilon_i: {self.epsilon_i}, epsilon_f: {self.epsilon_f}, epsilon_anneal_steps: {self.epsilon_anneal_steps}, max_steps: {max_steps}\n")
      print('===========================================================================================\n')
      
      checkpoint = self.checkpoint_every

      for ep in pbar:
        if total_steps > max_steps:
            break

        # Observar estado inicial como indica el algoritmo
        state, _ = self.env.reset()
        state_phi = self.state_processing_function(state)
        current_episode_reward = 0.0
        current_episode_steps = 0
        current_episode_actions = []
        done = False
       
       

        # Bucle principal de pasos dentro de un episodio
        for _ in range(max_steps):

            # Seleccionar acción epsilon-greedy usando select_action()
            action = self.select_action(state_phi, total_steps, train=True)
            current_episode_actions.append(action)

            # Ejecutar action = env.step(action)
            next_state, reward, terminated, truncated, _, = self.env.step(action)
            done = terminated or truncated

            # Procesar next_state con state_processing_function
            next_state_phi = self.state_processing_function(next_state)

            # Acumular reward y actualizar total_steps, current_episode_steps
            current_episode_reward += reward
            total_steps += 1
            current_episode_steps += 1

            # Almacenar transición en replay memory
            self.memory.add(state_phi, action, reward, done, next_state_phi)

            # Llamar a update_weights() para entrenar modelo
            self.update_weights()

            # Actualizar state y state_phi al siguiente estado
            state = next_state
            state_phi = next_state_phi

            # Comprobar condición de done o límite de pasos de episodio y break
            if done:
                break
            if current_episode_steps >= max_steps_episode:
                print(f"\nSe alcanzo {current_episode_steps} pasos en un mismo episodio.")
                break
            if total_steps > max_steps:
                print(f"\nEntrenamiento detenido: se alcanzaron {total_steps} pasos.")
                break 

        # Guardar datos para graficar
        loss = getattr(self, "last_loss", 0.0)
        counter = Counter(current_episode_actions)
        action_distribution = [counter.get(i, 0) for i in range(self.env.action_space.n)]
        self.all_actions.append(action_distribution)

        reward = np.mean(rewards[-self.episode_block:])
        mean_rewards.append(reward)

        # Implementación de epsilon adaptativo
        if self.adaptive_epsilon:
            if reward > self.best_reward:
                print(f"Reduciendo epsilon: Recompensa actual: {reward}, Epsilon: {self.epsilon_i}, Total steps: {total_steps}")
                self.best_reward = reward
                self.no_improvement_episodes = 0
                # Reduce epsilon (más explotación)
                self.epsilon_i = max(self.epsilon_i - self.epsilon_decrease, self.epsilon_min)
            else:
                print(f"Aumentando epsilon: Recompensa actual: {reward}, Epsilon: {self.epsilon_i}, Total steps: {total_steps}")
                self.no_improvement_episodes += 1
                if self.no_improvement_episodes >= self.patience:
                    # Aumenta epsilon (más exploración)
                    self.epsilon_i = min(self.epsilon_i + self.epsilon_increase, self.epsilon_max)
                    self.no_improvement_episodes = 0
        
        epsilon = self.compute_epsilon(total_steps)

        # Registro de métricas y progreso
        rewards.append(current_episode_reward)
        epsilons.append(epsilon)
        steps_per_episode.append(current_episode_steps)
        losses.append(loss)
        metrics["reward"] = reward
        metrics["epsilon"] = epsilon
        metrics["steps"] = total_steps
        pbar.set_postfix(metrics)

        isDQN = hasattr(self, "policy_net") and self.policy_net is not None
        if total_steps >= checkpoint:
            checkpoint += self.checkpoint_every
            print(f"\n=== Recompensa actual: {reward}, Epsilon: {epsilon}, Total steps: {total_steps} ===")
            
            
            if self.load_checkpoint:
                print(f"Checkpoint guardado en GenericDQNAgent-steps:{total_steps}-e:{epsilon}.dat")
                if isDQN:
                    os.makedirs(DQN_BREAKPOINT_DIR, exist_ok=True)
                    torch.save(self.policy_net.state_dict(), f"{DQN_BREAKPOINT_DIR}/GenericDQNAgent-run-{self.run_name}-steps-{total_steps}-e-{epsilon:.4f}-max_r-{reward}.dat")
                else:
                    os.makedirs(DDQN_BREAKPOINT_DIR, exist_ok=True)
                    torch.save(self.online_net.state_dict(), f"{DDQN_BREAKPOINT_DIR}/GenericDDQNAgent-run-{self.run_name}-steps-{total_steps}-e-{epsilon:.4f}-max_r-{reward}.dat")
           

      # Guardar el modelo entrenado  
      genericDataPath = getGenericDataFilePath(isDQN, self.run_name)
      if isDQN:
        # torch.save(self.policy_net.state_dict(), f"{model_path}")
        os.makedirs(DQN_COMMON_MODEL_PATH, exist_ok=True)
        torch.save(self.policy_net.state_dict(), genericDataPath)
      else:
        os.makedirs(DDQN_NET_HISTORY_DIR, exist_ok=True)
        torch.save(self.online_net.state_dict(), genericDataPath)

      # Guardar las métricas de entrenamiento

      # Crear carpeta si no existe
      metrics_dir = getMetricsDir(isDQN)
      os.makedirs(metrics_dir, exist_ok=True)

      # Guardar archivo con nombre personalizado dentro de esa carpeta
      savePath = getMetricFilePath(isDQN, self.run_name)
      np.savez(savePath,
         rewards=np.array(mean_rewards),
         losses=np.array(losses),
         actions=np.array(self.all_actions),
         steps=np.array(steps_per_episode),
         epsilons=np.array(epsilons))
      return rewards
    
        
    def compute_epsilon(self, steps_so_far):
        """
        Compute el valor de epsilon a partir del número de pasos dados hasta ahora.
        """
        if steps_so_far < self.epsilon_anneal_steps:
            epsilon = self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (steps_so_far / self.epsilon_anneal_steps)
        else:
            epsilon = self.epsilon_f
        return epsilon
    def play(self, env, episodes=1):
        """
        Modo evaluación: ejecutar episodios sin actualizar la red.
        """
        rewards = []
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            current_episode_reward = 0.0
            while not done:
                # Procesar el estado actual
                state_phi = self.state_processing_function(state)

                # Seleccionar acción greedy
                # Nota: se asume que select_action maneja train=False para seleccionar la acción greedy
                action = self.select_action(state_phi, self.total_steps, train=False)

                # ejecutar acción y actualizar estado
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                current_episode_reward += reward
            print(f"Recompensa total del episodio {ep}: {current_episode_reward}")
            rewards.append(current_episode_reward)
        print(f"Recompensa total promedio: {np.mean(rewards)}")

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        """
        Selecciona una acción a partir del estado actual. Si train=False, se selecciona la acción greedy.
        Si train=True, se selecciona la acción epsilon-greedy.
        
        Args:
            state: El estado actual del entorno.
            current_steps: El número de pasos actuales. Determina el valor de epsilon.
            train: Si True, se selecciona la acción epsilon-greedy. Si False, se selecciona la acción greedy.
        """
        pass

    @abstractmethod
    def update_weights(self):
        pass