import torch
import torch.nn as nn
import numpy as np
from abstract_agent import Agent
from replay_memory import ReplayMemory, Transition
import random


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        model,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_steps,
        episode_block,
        device,
    ):
        super().__init__(
            env,
            obs_processing_func,
            memory_buffer_size,
            batch_size,
            learning_rate,
            gamma,
            epsilon_i,
            epsilon_f,
            epsilon_anneal_steps,
            episode_block,
            device,
        )
        # Guardar entorno y función de preprocesamiento
        self.env = env
        self.obs_processing_func = obs_processing_func
        # Inicializar policy_net en device
        self.policy_net = model.to(self.device)
        # Configurar función de pérdida MSE y optimizador Adam
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )
        self.loss_fn = nn.MSELoss()
        # Crear replay memory de tamaño buffer_size
        self.memory = ReplayMemory(memory_buffer_size)
        # Almacenar batch_size, gamma y parámetros de epsilon-greedy
        # TODO no se si se refiere a guardar con "almacenar"
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.episode_block = episode_block

    def select_action(self, state, current_steps, train=True):
        # Calcular epsilon según step
        # Epsilon decrece con más pasos para pasar de exploración a explotación gradual
        epsilon = self.compute_epsilon(current_steps)

        # Durante entrenamiento: exploración aleatoria con probabilidad epsilon
        if train and random.random() < epsilon:
            return self.env.action_space.sample()
        
        # Explotación: seleccionar la acción greedy según Q-values
        # Si el estado ya es tensor, úsalo directamente; si es numpy, conviértelo
        if isinstance(state, torch.Tensor):
            state_tensor = state.to(self.device).float().unsqueeze(0)
        else:
            arr = np.asarray(state, dtype=np.float32)
            state_tensor = torch.from_numpy(arr).to(self.device).unsqueeze(0)
        
        # Con torch.no_grad() evitamos calcular gradientes, ya que no entrenamos en este paso
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        # greedy_action
        return q_values.argmax(dim=1).item()

    def update_weights(self):
        # 1) Comprobar que hay al menos batch_size muestras en memoria
        # Evitar entrenar con pocos datos que causen actualizaciones ruidosas
        if len(self.memory) < self.batch_size:
            return

        # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
        # El muestreo aleatorio reduce correlaciones y estabiliza el aprendizaje
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Armar batch de estados
        states = torch.stack(batch.state).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)

        # Convertir acciones, recompensas y dones a tensores
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # 3) Calcular q_current con policy_net(states).gather(...)
        # gather extrae el Q-value correspondiente a la acción tomada en cada muestra.
        q_current = self.policy_net(states).gather(1, actions)

        # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
        # No computar gradientes aquí para mantener la estabilidad de los objetivos
        with torch.no_grad():
            max_q_next = self.policy_net(next_states).max(dim=1, keepdim=True).values
            max_q_next = max_q_next * (1 - dones)   

        # 5) Calcular target = rewards + gamma * max_q_next_state
        # Objetivo de Bellman: recompensa inmediata + valor descontado del siguiente estado
        q_target = rewards + self.gamma * max_q_next

        # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
        # Minimizar esta pérdida ajusta la red para aproximar la función Q óptima
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Clipping de gradientes podría añadirse aquí para mayor estabilidad
        self.optimizer.step()
