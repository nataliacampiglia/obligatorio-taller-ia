# Memoria de Repetición Priorizada

Este documento describe la implementación de memoria de repetición priorizada (Prioritized Experience Replay) que se ha agregado al proyecto, manteniendo la compatibilidad con la memoria de repetición regular existente.

## Características

- **Memoria de Repetición Regular**: Mantiene la funcionalidad original
- **Memoria de Repetición Priorizada**: Nueva implementación que prioriza transiciones según su error TD
- **Compatibilidad**: Ambas memorias pueden usarse intercambiablemente mediante un flag
- **Factory Pattern**: Clase factory para crear diferentes tipos de memoria

## Implementación

### Clases Principales

#### 1. `PrioritizedReplayMemory`
```python
class PrioritizedReplayMemory:
    def __init__(self, capacity=4, device=None, alpha=0.6, beta=0.4, 
                 beta_increment=0.001, epsilon=1e-6):
```

**Parámetros:**
- `capacity`: Capacidad máxima de la memoria
- `device`: Dispositivo para tensores (CPU/GPU)
- `alpha`: Parámetro de priorización (0 = uniforme, 1 = completamente priorizado)
- `beta`: Parámetro de importancia sampling (0 = sin corrección, 1 = corrección completa)
- `beta_increment`: Incremento de beta por actualización
- `epsilon`: Valor pequeño para evitar prioridades cero

#### 2. `ReplayMemoryFactory`
```python
class ReplayMemoryFactory:
    @staticmethod
    def create_memory(memory_type="regular", **kwargs):
```

**Tipos de memoria:**
- `"regular"`: Memoria de repetición uniforme
- `"prioritized"`: Memoria de repetición priorizada

## Uso

### 1. Con DQN Agent

#### Memoria Regular (por defecto)
```python
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
    run_name="regular_example",
    use_prioritized_replay=False  # Memoria regular
)
```

#### Memoria Priorizada
```python
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
    run_name="prioritized_example",
    use_prioritized_replay=True,  # Memoria priorizada
    prioritized_replay_alpha=0.6,  # Parámetro de priorización
    prioritized_replay_beta=0.4,   # Parámetro de importancia sampling
    prioritized_replay_beta_increment=0.001,  # Incremento de beta
    prioritized_replay_epsilon=1e-6  # Valor pequeño para evitar prioridades cero
)
```

### 2. Con Double DQN Agent

```python
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
```

### 3. Usando el Factory

```python
from replay_memory import ReplayMemoryFactory

# Crear memoria regular
regular_memory = ReplayMemoryFactory.create_memory(
    memory_type="regular",
    capacity=1000,
    device=torch.device("cpu")
)

# Crear memoria priorizada
prioritized_memory = ReplayMemoryFactory.create_memory(
    memory_type="prioritized",
    capacity=1000,
    device=torch.device("cpu"),
    alpha=0.6,
    beta=0.4
)
```

## Diferencias en el Comportamiento

### Memoria Regular
- Muestreo uniforme de transiciones
- No requiere actualización de prioridades
- Loss estándar sin pesos de importancia sampling

### Memoria Priorizada
- Muestreo basado en prioridades (error TD)
- Actualización automática de prioridades después de cada entrenamiento
- Loss con pesos de importancia sampling para corregir el sesgo
- Parámetros configurables para ajustar el comportamiento

## Parámetros de la Memoria Priorizada

### Alpha (α)
- Controla el grado de priorización
- α = 0: Muestreo uniforme (equivalente a memoria regular)
- α = 1: Muestreo completamente priorizado
- Valores típicos: 0.4 - 0.6

### Beta (β)
- Controla la corrección de importancia sampling
- β = 0: Sin corrección (puede causar sesgo)
- β = 1: Corrección completa
- Se incrementa gradualmente durante el entrenamiento
- Valores típicos: 0.4 - 0.6 inicial, incrementando a 1.0

### Epsilon (ε)
- Valor pequeño agregado a las prioridades
- Evita que las prioridades sean exactamente cero
- Valores típicos: 1e-6

## Ventajas de la Memoria Priorizada

1. **Aprendizaje más eficiente**: Las transiciones con mayor error TD se muestrean más frecuentemente
2. **Mejor convergencia**: Enfoca el aprendizaje en las experiencias más informativas
3. **Reducción del tiempo de entrenamiento**: Menos episodios necesarios para alcanzar el mismo rendimiento

## Consideraciones

1. **Complejidad computacional**: La memoria priorizada requiere más cálculos
2. **Hiperparámetros adicionales**: Necesita ajuste de α, β y ε
3. **Memoria adicional**: Almacena prioridades además de las transiciones

## Ejemplo Completo

Ver el archivo `example_usage.py` para ejemplos completos de uso de ambos tipos de memoria.

## Compatibilidad

La implementación es completamente compatible con el código existente:
- El flag `use_prioritized_replay=False` (por defecto) mantiene el comportamiento original
- No se requieren cambios en el código existente
- Ambos agentes (DQN y Double DQN) soportan ambos tipos de memoria 