import random
from collections import namedtuple

# Lo primero que se define es la tupla Transition para almacenar cada experiencia.
# Por qué: facilita el acceso por nombre a los diferentes componentes de la transición.
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, capacity):
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

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        # Creamos una instancia de Transition con los datos pasados.
        # Por qué: agrupa los elementos de la experiencia en un solo objeto inmutable.
        transition = Transition(state, action, reward, done, next_state)

        # Verificamos si la memoria no está llena.
        if len(self.memory) < self.capacity:
            # Si hay espacio, simplemente añadimos, conservando todas las experiencias.
            self.memory.append(transition)
        else:
            # Si ya está llena, reemplazamos la transición en la posición actual, eliminando la más antigua y manteniendo la memoria actualizada.
            self.memory[self.position] = transition
        
        # Avanzamos el puntero; al llegar a capacity volvemos a 0
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
      """
      Devuelve un batch aleatorio de transiciones.
      Params:
      - batch_size (int): número de transiciones a muestrear.
      Returns:
      - lista de Transition de longitud batch_size.
      """
      # Validamos que batch_size no supere el número de elementos almacenados, para evitar errores al muestrear. 
      if batch_size > len(self.memory):
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
