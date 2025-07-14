import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abstract_agent import Agent
from replay_memory import ReplayMemory, PrioritizedReplayMemory, Transition
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
        run_name="dqn_run",
        use_prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        prioritized_replay_beta_increment=0.001,
        prioritized_replay_epsilon=1e-6,
        adaptive_epsilon=False,
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
            run_name="dqn_run",
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
        # TO TRY: otra función de error
        self.loss_fn = nn.MSELoss()
        # Loss function para memoria priorizada (sin reducción)
        self.loss_fn_none = nn.MSELoss(reduction='none')
        
        # Configurar tipo de memoria de repetición
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(
                capacity=memory_buffer_size,
                device=device,
                alpha=prioritized_replay_alpha,
                beta=prioritized_replay_beta,
                beta_increment=prioritized_replay_beta_increment,
                epsilon=prioritized_replay_epsilon
            )
        else:
            self.memory = ReplayMemory(memory_buffer_size)
            
        # Almacenar batch_size, gamma y parámetros de epsilon-greedy
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.episode_block = episode_block
        self.run_name = run_name
        self.adaptive_epsilon = adaptive_epsilon

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
            # print(f"Q-values: {q_values}")  # Debugging line to check Q-values
        # greedy_action
        return q_values.argmax(dim=1).item()

    def update_weights(self):
        # 1) Comprobar que hay al menos batch_size muestras en memoria
        # Evitar entrenar con pocos datos que causen actualizaciones ruidosas
        if len(self.memory) < self.batch_size:
            return

        # 2) Muestrear minibatch según el tipo de memoria
        if self.use_prioritized_replay:
            # Muestreo priorizado
            transitions, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Muestreo uniforme
            transitions = self.memory.sample(self.batch_size)
            indices = None
            weights = None

        batch = Transition(*zip(*transitions))
        # states, actions, reward, next_state, done = zip(transitions*)

        # Armar batch de estados
        states = torch.stack(batch.state).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        # states_t (tensor) y next_state_t => shape = (batch_size=32, 4, 84,84)

        # Convertir acciones, recompensas y dones a tensores
        # actions_t, rewards_t, dones_t => shape = (batch_size, 1)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # 3) Calcular q_current con policy_net(states).gather(...)
        # gather extrae el Q-value correspondiente a la acción tomada en cada muestra.
        q_current = self.policy_net(states).gather(1, actions)

        # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
        # No computar gradientes aquí para mantener la estabilidad de los objetivos
        with torch.no_grad():
            max_q_next = self.policy_net(next_states).max(dim=1, keepdim=True).values  # bx1
            max_q_next = max_q_next * (1 - dones)  # si es el ultimo vale 0

        # 5) Calcular target = rewards + gamma * max_q_next_state
        # Objetivo de Bellman: recompensa inmediata + valor descontado del siguiente estado
        q_target = rewards + self.gamma * max_q_next

        # 6) Computar loss según el tipo de memoria
        if self.use_prioritized_replay:
            with torch.no_grad():
                q_target_cpu = q_target.detach().cpu()
                q_current_cpu = q_current.detach().cpu()
                td_errors = torch.abs(q_target_cpu - q_current_cpu).numpy().flatten()


            # Usar loss_fn_none para obtener loss sin reducción
            loss_per_sample = self.loss_fn_none(q_current, q_target).squeeze()
            loss = (weights * loss_per_sample).mean()

            # Actualizar prioridades
            self.memory.update_priorities(indices, td_errors)

            ### liberar tensores intermedios
            del td_errors, loss_per_sample, weights
            torch.cuda.empty_cache() 
        else:
            # Loss estándar para memoria regular
            loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clipping de gradientes podría añadirse aquí para mayor estabilidad
        self.optimizer.step()

        # Guardar el último valor de pérdida para poder graficarlo luego
        self.last_loss = loss.item()

