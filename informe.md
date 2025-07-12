# Funciones:

A continuacion describiremos las funciones a utilizar

* **Phi:** La función para procesar los estados (phi en el paper) que es necesaria para poder usar el modelo de Pytorch con las representaciones de gym. Esta función pasa una observación de gym a un tensor de Pytorch y la normaliza. En este trabajo, la función process_state implementa el rol de la función phi descrita en el paper. Su objetivo es procesar los estados (observaciones) obtenidos del entorno de gym para que puedan ser utilizados como entrada del modelo de PyTorch. Específicamente, toma una observación de gym, la convierte en un tensor de PyTorch de tipo float32, la transfiere al dispositivo adecuado (CPU o GPU) y normaliza sus valores dividiéndolos por 255.0. Esta normalización es especialmente útil cuando se trabaja con imágenes o datos en formato de píxeles.
* **create_env:** Crea y configura el entorno de gym con los parámetros necesarios para el entrenamiento y evaluación, incluyendo la grabación de videos y el preprocesamiento de observaciones.
* **DQN_CNN_Model:** Implementa la arquitectura de red neuronal convolucional según Mnih et al. (2013) para el algoritmo DQN. La arquitectura incluye dos capas convolucionales (16 filtros 8×8/4 y 32 filtros 4×4/2), una capa fully-connected intermedia de 256 unidades, y una capa de salida que produce un Q-value por cada acción posible. Esta red procesa observaciones del entorno (imágenes) y estima los valores Q para guiar la política del agente.
* **load_dqn_agent:** Crea la policy network usando DQN_CNN_Model, carga los pesos de una red previamente entrenada si se especifica, y crea el agente DQN con la red configurada y los hiperparámetros necesarios para su funcionamiento.
* **create_reference_states:** Genera un conjunto de estados de referencia a partir del entorno, útiles para evaluar la convergencia y el desempeño del agente en diferentes etapas del entrenamiento.
* **save_q_values:** Calcula y guarda los valores Q estimados por la red para un conjunto de estados de referencia, almacenándolos en un archivo para su posterior análisis.
* **execute_dqn_training_phase:** Ejecuta una fase de entrenamiento para un agente DQN, entrenando la red, guardando los valores Q y devolviendo el agente entrenado.
* **execute_ddqn_training_phase:** Ejecuta una fase de entrenamiento para un agente Double DQN (DDQN), entrenando la red, guardando los valores Q y devolviendo el agente entrenado.
* **execute_agent_play:** Permite ejecutar un agente entrenado en modo evaluación, grabando un video de su desempeño en el entorno.
* **getVideoFolder:** Devuelve la ruta de la carpeta donde se almacenan los videos de validación para una fase y tipo de agente determinados.
* **getVideoPath:** Devuelve la ruta completa al archivo de video generado durante la validación para una fase y tipo de agente determinados.

## Agente

Se defin la clase agente (abstracto), encargado de interactuar con el ambiente y entrenar los modelos. Los métdos definidos deben funcionar para ambos problemas simplemente cambiando el modelo a utilizar para cada ambiente.  El agente será definido usando las funciones `load_dqn_agent` y `execute_ddqn_training_phase` para los modelos DQN y DDQN respectivamente.

### Funciones implementadas:

1. **`__init__`**: Inicializa los parámetros del agente, incluyendo la función de procesamiento de estados (phi), configuración de la replay memory que podra ser normal o priorizada, asignación de hiperparámetros de aprendizaje (gamma, epsilon, learning rate), y los parámetros de entrenamiento (batch size, episode block, checkpoint frequency).

2. **`compute_epsilon`**: Calcula el valor actual de epsilon basado en el número de pasos transcurridos. Implementa un decaimiento lineal desde epsilon inicial hasta epsilon final durante los pasos de annealing, manteniendo el valor final después de alcanzar el límite.

3. **`select_action`**: Método abstracto que debe ser implementado por las subclases. Selecciona acciones de forma epsilon-greedy durante el entrenamiento (train=True) y de forma completamente greedy durante la evaluación (train=False).

4. **`train`**: Entrena el agente por un número dado de episodios con longitud determinada. Implementa el bucle principal de entrenamiento: selección de acciones, ejecución en el entorno, almacenamiento de transiciones en memoria, actualización de pesos, y registro de métricas. Incluye funcionalidad de checkpoint y guardado de modelos.

5. **`play`**: Permite grabar episodios de evaluación donde el agente siempre selecciona la mejor acción conocida (modo greedy). No actualiza los pesos de la red durante la ejecución.

6. **`update_weights`**: Método abstracto que debe ser implementado por las subclases para definir la lógica específica de actualización de pesos según el algoritmo DQN o DDQN.









