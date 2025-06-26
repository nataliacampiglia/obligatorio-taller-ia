import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device, sync_target = 1000, run_name="dqn_run", use_clip=True):
        
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
        self.episode_block = episode_block
        self.sync_target = sync_target  # Store the sync_target parameter
        self.run_name = run_name

        # Inicializar contador de pasos para sincronizar target
        self.sync_counter = 0  # Start at 0, will sync after sync_target steps

        self.use_clip = use_clip

    
    def select_action(self, state, current_steps, train=True):
      # Calcular epsilon decay según step (entre eps_start y eps_end en eps_steps)
      epsilon = self.compute_epsilon(current_steps)

      # Si train y con probabilidad epsilon: acción aleatoria
      if train and random.random() < epsilon:
          return self.env.action_space.sample()
      
      
      if isinstance(state, torch.Tensor):
          state_tensor = state.to(self.device).float().unsqueeze(0)
      else:
          # torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
          arr = np.asarray(state, dtype=np.float32)
          state_tensor = torch.from_numpy(arr).to(self.device).unsqueeze(0)
       
      with torch.no_grad():
        q_values = self.online_net(state_tensor)
      # greedy_action
      return q_values.argmax(dim=1).item()

    
    def update_weights(self):
      # 1) Verificar que haya al menos batch_size transiciones en memoria
      if len(self.memory) < self.batch_size:
        return
      
      # 2) Muestrear minibatch y convertir estados, acciones, recompensas, dones y next_states a tensores
      transitions = self.memory.sample(self.batch_size)
      batch = Transition(*zip(*transitions))

      states = torch.stack(batch.state).to(self.device)
      next_states = torch.stack(batch.next_state).to(self.device)

      actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
      rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
      dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

      # 3) Calcular q_current: online_net(states).gather(…)
      q_current = self.online_net(states).gather(1, actions)
      
      # 4) Calcular target Double DQN:
      # a) Elegimos las mejores acciones siguientes usando la red online (max_a' Q_online(s',a'))
      with torch.no_grad():
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        # b) Evaluamos esas acciones en la red target (Q_target(s', a'))
        q_next = self.target_net(next_states).gather(1, next_actions)
        q_next = q_next * (1 - dones)
       
     # c) Calculamos el target: r + gamma * Q_target * (1 - done)
      target_q = rewards + self.gamma * q_next
      # 5) Computar loss MSE entre q_current y target_q, backprop y optimizer.step()
      loss = self.loss_fn(q_current, target_q)
      self.optimizer.zero_grad()
      loss.backward()
      
      # Optional: Add gradient clipping for stability
      if self.use_clip:
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
      
      self.optimizer.step()
      self.last_loss = loss.item()

      # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
      self.sync_counter += 1
      if self.sync_counter >= self.sync_target:
        # Cada sync_target pasos, copiamos los pesos de la red online a la target
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.sync_counter = 0  # Reset counter after syncing