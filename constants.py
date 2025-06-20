import torch

SEED = 23

# Entorno - https://ale.farama.org/environments/breakout/
ENV_NAME = "ALE/Breakout-v5" 

# Vemos que dispositivo tenemos, si es GPU, MPS o CPU. **El uso de GPU es altamente recomendable** para acelerar el entrenamiento de los modelos. 
# Detectar el mejor dispositivo disponible
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ENV ATTRIBUTES
GRAY_SCALE = True # si True, convertimos la imagen a escala de grises
SCREEN_SIZE = 84 # redimensionamos a SCREEN_SIZExSCREEN_SIZE
NUM_STACKED_FRAMES = 4 # apilamos NUM_STACKED_FRAMES frames
SKIP_FRAMES = 4 # saltamos SKIP_FRAMES frames (haciendo la misma acción)


######################################################
#                   Entrenamiento                    # 
#  Hiperparámetros de entrenamiento del agente DQN   #
######################################################
# TOTAL_STEPS = 10_000_000
TOTAL_STEPS = 2_000_000
EPISODES = 10_000
STEPS_PER_EPISODE = 20_000

EPSILON_INI = 1
EPSILON_MIN = 0.05
# EPSILON_ANNEAL_STEPS = 1_000_000
EPSILON_ANNEAL_STEPS = 700_000

EPISODE_BLOCK = 100

BATCH_SIZE = 32
BUFFER_SIZE = 50_000

GAMMA = 0.995
LEARNING_RATE = 1e-5


# Ruta del modelo guardado
METRICS_DIR = "metrics"
COMMON_METRICS_PATH = f"{METRICS_DIR}/metrics_"
COMMON_MODEL_PATH = "net_history/GenericDQNAgent_" 
MODEL_PATH = "net_history/GenericDQNAgent.dat"  # Cambiar si se usa timestamp
PHASE1_MODEL_PATH = "net_history/GenericDQNAgent_phase1.dat" 
