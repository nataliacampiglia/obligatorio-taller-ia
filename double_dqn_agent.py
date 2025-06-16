import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device, sync_target = 1000, run_name="dqn_run",):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device, run_name)
        # Guardar entorno y función de preprocesamiento
        self.env = gym_env
        self.obs_processing_func = obs_processing_func
        # Inicializar online_net (model_a) y target_net (model_b) en device
        self.online_net = model_a.to(device)
        self.target_net = model_b.to(device)

        # Sincronizar target_net con online_net al inicio para evitar que target_net esté desincronizado y los primeros pasos de entrenamiento no sean erráticos
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Asegurarse de que target_net esté en modo evaluación


        # Configurar función de pérdida MSE y optimizador Adam para online_net
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
      
        # Crear replay memory de tamaño buffer_size
        self.memory = ReplayMemory(memory_buffer_size)

        # Almacenar batch_size, gamma, parámetros de epsilon y sync_target
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.run_name = run_name

        # Inicializar contador de pasos para sincronizar target
        self.sync_counter = sync_target
        pass
    
    def select_action(self, state, current_steps, train=True):
      # Calcular epsilon decay según step (entre eps_start y eps_end en eps_steps)
      epsilon = self.compute_epsilon(current_steps)

      # Si train y con probabilidad epsilon: acción aleatoria
      if train and random.random() < epsilon:
          return self.env.action_space.sample()
      
      with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.online_net(state)
        return q_values.argmax().item()
    
    def update_weights(self):
      # 1) Verificar que haya al menos batch_size transiciones en memoria
      if len(self.memory) < self.batch_size:
        return
      
      # 2) Muestrear minibatch y convertir estados, acciones, recompensas, dones y next_states a tensores
      transitions = self.memory.sample(self.batch_size)
      batch = Transition(*zip(*transitions))

      states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
      actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
      rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
      next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
      dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

      # 3) Calcular q_current: online_net(states).gather(…)
      q_current = self.online_net(states).gather(1, actions)
      
      # 4) Calcular target Double DQN:
      # a) Elegimos las mejores acciones siguientes usando la red online (max_a' Q_online(s',a'))
      with torch.no_grad():
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        # b) Evaluamos esas acciones en la red target (Q_target(s', a'))
        q_next = self.target_net(next_states).gather(1, next_actions)
        # c) Calculamos el target: r + gamma * Q_target * (1 - done)
        target_q = rewards + self.gamma * q_next * (1 - dones)

      # 5) Computar loss MSE entre q_current y target_q, backprop y optimizer.step()
      loss = self.loss_fn(q_current, target_q)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
      self.sync_counter -= 1
      if self.sync_counter == 0:
        # Cada sync_target pasos, copiamos los pesos de la red online a la target
        self.target_net.load_state_dict(self.online_net.state_dict())
      
            