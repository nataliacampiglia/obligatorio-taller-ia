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

# Tipos de Agentes DQN 
DQN_TYPE = "dqn"
DOUBLE_DQN_TYPE = "ddqn"


# Ruta del modelo guardado
METRICS_DIR = "metrics"
DQN_METRICS_DIR = 'metrics/dqn'
DDQN_METRICS_DIR = 'metrics/ddqn'
COMMON_METRICS_PATH = f"{METRICS_DIR}/metrics_"
DQN_COMMON_METRICS_PATH = f"{DQN_METRICS_DIR}/metrics_"
DDQN_COMMON_METRICS_PATH = f"{DDQN_METRICS_DIR}/metrics_"
NET_HISTORY_DIR = "net_history"
DQN_NET_HISTORY_DIR = f"{NET_HISTORY_DIR}/dqn"
DDQN_NET_HISTORY_DIR = f"{NET_HISTORY_DIR}/ddqn"
BREAKPOINT_DIR = "breakpoints"
DQN_BREAKPOINT_DIR = f"{BREAKPOINT_DIR}/dqn"
DDQN_BREAKPOINT_DIR = f"{BREAKPOINT_DIR}/ddqn"

DQN_COMMON_MODEL_PATH = F"{DQN_NET_HISTORY_DIR}/GenericDQNAgent-" 
DDQN_COMMON_MODEL_PATH = F"{DDQN_NET_HISTORY_DIR}/GenericDDQNAgent-"
MODEL_PATH = "net_history/GenericDQNAgent.dat"  # Cambiar si se usa timestamp
PHASE1_MODEL_PATH = "net_history/GenericDQNAgent_phase1.dat" 

# Epsilon adaptativo
EPSILON_ADAPTIVE_PATIENCE = 40
EPSILON_ADAPTIVE_INCREASE = 1.05   # multiplicativo
EPSILON_ADAPTIVE_DECREASE = 0.90  # multiplicativo
IMPROVEMENT_THRESHOLD = 0.5


def getMetricsDir(isDQN):
    if isDQN:
        return DQN_METRICS_DIR
    return DDQN_METRICS_DIR

def getCommonMetricFilePath(isDQN ):
    dirPath = getMetricsDir(isDQN)
    return f"{dirPath}/metrics_"


def getMetricFilePath(isDQN, run_name):
    filePath = getCommonMetricFilePath(isDQN)
    return f"{filePath}{run_name}.npz"

def getGenericDataDir(isDQN):
    if isDQN:
        return DQN_NET_HISTORY_DIR
    return DDQN_NET_HISTORY_DIR

def getCommonDataFilePath(isDQN):
    dirPath = getGenericDataDir(isDQN)
    return f"{dirPath}/{f'GenericDQNAgent-' if isDQN else 'GenericDDQNAgent-'}"

def getGenericDataFilePath(isDQN, run_name):
    filePath = getCommonDataFilePath(isDQN)
    return f"{filePath}{run_name}.dat"

def getMetricFilePathList(isDQN, phase_ids):
    filesPath = []
    print(phase_ids)
    for path_id in phase_ids:
        filePath = getMetricFilePath(isDQN, path_id)
        filesPath.append(filePath)
    return filesPath