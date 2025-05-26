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
        # Durante entrenamiento: con probabilidad epsilon acción aleatoria
        #                   sino greedy_action
        # Durante evaluación: usar greedy_action (o pequeña epsilon fija)
        # si no es training epsilon sera 0, entonces elegira funcio greedy
        epsilon = self.compute_epsilon(current_steps) if train else 0.0

        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            best_Action = q_values.argmax(dim=1).item()
            # print(f"{ best_Action = }")
            return best_Action

    def update_weights(self):
        # import gc
        # gc.collect()
        # if torch.backends.mps.is_available():
        #       torch.mps.empty_cache()
        # 1) Comprobar que hay al menos batch_size muestras en memoria

        if len(self.memory) < self.batch_size:
            return

        # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(self.device)

        # print(f"{ states.dtype = }")
        # print(f"{ states.shape = }")
        # print(f"{ states.size() = }")
        # states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        # next_states = torch.FloatTensor(batch.next_state).to(self.device)

        # 3) Calcular q_current con policy_net(states).gather(...)
        q_current = self.policy_net(states).gather(1, actions)

        # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
        with torch.no_grad():
            max_q_next_state = self.policy_net(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
            # 5) Calcular target = rewards + gamma * max_q_next_state
            # q_target = rewards + self.gamma * max_q_next_state * (1 - dones)
            #q_target = rewards + self.gamma * max_q_next_state
            q_target = rewards + self.gamma * max_q_next_state

        # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
