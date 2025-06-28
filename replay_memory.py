import random
import torch
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, capacity=4, device=None):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
        - capacity (int): número máximo de transiciones a almacenar.
        """
        # Guardamos la capacidad máxima de la memoria para controlar el tamaño de la memoria y evitar que crezca sin control.
        self.capacity = capacity

        # Inicializamos la lista que contendrá las transiciones. Se inicializa como un array vacío para permitir el append 
        # y el remplazo de la memoria mediante un índice.
        self.memory = []

        # Creamos un puntero circular position inicializado en 0.
        # Por qué: indica la posición donde se sobrescribirá la próxima transición cuando esté llena.
        self.position = 0

        self.device = device
        

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        # Creamos una instancia de Transition con los datos pasados.
        # Por qué: agrupa los elementos de la experiencia en un solo objeto inmutable.

        transition = Transition(state, action, reward, done, next_state)
        if not isinstance(state, torch.Tensor):
          print("WARNING: State no es un tensor, se convertirá a tensor.")
          state = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)
        # Verificamos si la memoria no está llena.
        if len(self.memory) < self.capacity:
            # Si hay espacio, simplemente añadimos, conservando todas las experiencias.
            self.memory.append(transition)
        else:
            # Si ya está llena, reemplazamos la transición en la posición actual, eliminando la más antigua y manteniendo la memoria actualizada.
            self.memory[self.position] = transition
        
        # Avanzamos el puntero; al llegar a capacity volvemos a 0
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=32):
      """
      Devuelve un batch aleatorio de transiciones.
      Params:
       - batch_size (int): número de transiciones a muestrear.
      Returns:
       - lista de Transition de longitud batch_size.
      """
      # Validamos que batch_size no supere el número de elementos almacenados, para evitar errores al muestrear. 
      if batch_size > len(self):
        raise ValueError(f"Batch size {batch_size} mayor que memoria actual {len(self.memory)}")
      
      # Retornamos una muestra aleatoria sin reemplazo. Random.sample asegura que no haya repeticiones en el batch
      return random.sample(self.memory, batch_size)

      
    def __len__(self):
      """
      Devuelve el número actual de transiciones en memoria.
      """
      # Devolvemos el número de transiciones actualmente almacenadas. Útil para condicionar el inicio del entrenamiento 
      # cuando haya suficientes muestras.
      return len(self.memory)
    
    def clear(self):
      """
      Elimina todas las transiciones de la memoria.
      """
       # Limpiamos todas las transiciones almacenadas, lo que permite reiniciar la memoria al cambiar de tarea o retomar entrenamiento desde cero.
      self.memory.clear()

      # Reiniciamos el puntero position a 0, asegurando que las nuevas transiciones comiencen a guardar desde el inicio.
      self.position = 0


class SumTree:
    """
    Implementación de SumTree para muestreo priorizado eficiente.
    Permite muestreo en O(log n) en lugar de O(n).
    """
    
    def __init__(self, capacity):
        """
        Inicializa el SumTree.
        Params:
        - capacity: capacidad máxima del árbol
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.position = 0
        
    def _propagate(self, idx, change):
        """
        Propaga el cambio de prioridad hacia arriba en el árbol.
        """
        parent = (idx - 1) // 2
        
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """
        Recupera el índice de la hoja correspondiente al valor s.
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """
        Retorna la suma total de todas las prioridades.
        """
        return self.tree[0]
    
    def add(self, p, data):
        """
        Agrega un elemento con prioridad p y datos data.
        """
        idx = self.position + self.capacity - 1
        
        self.data[self.position] = data
        self.update(idx, p)
        
        self.position = (self.position + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        """
        Actualiza la prioridad del elemento en el índice idx.
        """
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        """
        Obtiene el elemento correspondiente al valor s.
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayMemory:
    """
    Implementación de memoria de repetición priorizada usando SumTree.
    Las transiciones se muestrean con probabilidad proporcional a su prioridad (error TD).
    Muestreo eficiente en O(log n) usando SumTree.
    """
    
    def __init__(self, capacity=4, device=None, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """
        Inicializa la memoria de repetición priorizada.
        Params:
        - capacity (int): número máximo de transiciones a almacenar
        - device: dispositivo para tensores
        - alpha (float): parámetro de priorización (0 = uniforme, 1 = completamente priorizado)
        - beta (float): parámetro de importancia sampling (0 = sin corrección, 1 = corrección completa)
        - beta_increment (float): incremento de beta por actualización
        - epsilon (float): valor pequeño para evitar prioridades cero
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # SumTree para muestreo eficiente
        self.sum_tree = SumTree(capacity)
        
        # Contador de elementos
        self.n_entries = 0
        
    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria con prioridad máxima.
        """
        transition = Transition(state, action, reward, done, next_state)
        
        # Asignar prioridad máxima para asegurar que se muestree al menos una vez
        max_priority = self.sum_tree.tree.max() if self.n_entries > 0 else 1.0
        
        self.sum_tree.add(max_priority, transition)
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
    def sample(self, batch_size=32):
        """
        Muestrea transiciones con probabilidad proporcional a su prioridad usando SumTree.
        Returns:
        - lista de Transition de longitud batch_size
        - índices de las transiciones muestreadas
        - pesos de importancia sampling
        """
        if batch_size > self.n_entries:
            raise ValueError(f"Batch size {batch_size} mayor que memoria actual {self.n_entries}")
            
        batch = []
        indices = []
        priorities = []
        
        # Calcular segmentos para muestreo uniforme en el rango de prioridades
        segment = self.sum_tree.total() / batch_size
        
        for i in range(batch_size):
            # Muestrear un valor uniforme en el segmento correspondiente
            s = random.uniform(segment * i, segment * (i + 1))
            
            # Obtener el elemento correspondiente del SumTree
            idx, priority, data = self.sum_tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calcular pesos de importancia sampling
        weights = np.array(priorities) ** (-self.beta)
        weights /= weights.max()  # Normalizar
        
        # Actualizar beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):
        """
        Actualiza las prioridades basándose en los errores TD.
        Params:
        - indices: índices de las transiciones
        - td_errors: errores TD correspondientes
        """
        for idx, td_error in zip(indices, td_errors):
            # Calcular nueva prioridad basada en el error TD
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.sum_tree.update(idx, priority)
            
    def __len__(self):
        """
        Devuelve el número actual de transiciones en memoria.
        """
        return self.n_entries
        
    def clear(self):
        """
        Elimina todas las transiciones de la memoria.
        """
        self.sum_tree = SumTree(self.capacity)
        self.n_entries = 0


class ReplayMemoryFactory:
    """
    Factory para crear diferentes tipos de memoria de repetición.
    """
    
    @staticmethod
    def create_memory(memory_type="regular", **kwargs):
        """
        Crea una instancia de memoria de repetición según el tipo especificado.
        Params:
        - memory_type (str): "regular" o "prioritized"
        - **kwargs: argumentos adicionales para la memoria
        """
        if memory_type == "regular":
            return ReplayMemory(**kwargs)
        elif memory_type == "prioritized":
            return PrioritizedReplayMemory(**kwargs)
        else:
            raise ValueError(f"Tipo de memoria no válido: {memory_type}. Use 'regular' o 'prioritized'")


