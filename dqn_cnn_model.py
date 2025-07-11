import torch.nn as nn
import torch.nn.functional as F

def conv2d_output_shape(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """
    Calcula (H_out, W_out) para una capa Conv2d con:
      - input_size: (H_in, W_in)
      - kernel_size, stride, padding, dilation: int o tupla (altura, ancho)
    Basado en:
      H_out = floor((H_in + 2*pad_h - dil_h*(ker_h−1) - 1) / str_h + 1)
      W_out = floor((W_in + 2*pad_w - dil_w*(ker_w−1) - 1) / str_w + 1)
    Fuente: Shape section en torch.nn.Conv2d :contentReference[oaicite:0]{index=0}
    """
    # Unifica todos los parámetros a tuplas (h, w)
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    H_in, W_in = input_size
    ker_h, ker_w = to_tuple(kernel_size)
    str_h, str_w = to_tuple(stride)
    pad_h, pad_w = to_tuple(padding)
    dil_h, dil_w = to_tuple(dilation)

    H_out = (H_in + 2*pad_h - dil_h*(ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2*pad_w - dil_w*(ker_w - 1) - 1) // str_w + 1

    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    def __init__(self,  obs_shape, n_actions):
        """
        CNN según Mnih et al. (2013):
        - 2 capas convolucionales (16 @ 8×8/4 y 32 @ 4×4/2)
        - 1 capa fully-connected intermedia de 256 unidades
        - 1 capa de salida con un Q-value por acción
        """
        super().__init__()

        # Desempaquetar canales y dimensiones
        in_channels, h, w = obs_shape
        
        # Primera capa conv: 16 filtros, kernel 8x8, stride 4. Los kernels grandes (8×8) con stride 4, capturan patrones espaciales gruesos
        # Impacto en la imagen: reduce la resolución de (h, w) a ((h-8)/4+1, (w-8)/4+1), filtrando detalles finos.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)

        # Segunda capa convolucional. Los kernels más pequeños (4×4) con stride 2 refina características detectadas
        # Impacto en la imagen: reduce aún más el tamaño espacial, enfocándose en estructuras intermedias.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Cálculo dinámico del tamaño tras convoluciones para aplanamiento
        h1, w1 = conv2d_output_shape((h, w), kernel_size=8, stride=4)
        h2, w2 = conv2d_output_shape((h1, w1), kernel_size=4, stride=2)

        flattened_size = 32 * h2 * w2

        # Capa fully-connected intermedia de 256 neuronas
        # Impacto: convierte mapas de características en un vector que sintetiza toda la información.
        self.fc = nn.Linear(flattened_size, 256)

        # Capa de salida con un Q-value por cada acción posible. Cada neurona estima el valor futuro esperado de tomar su acción correspondiente
        # Impacto: produce el vector de Q-values que guiará la política ε-greedy.
        self.out = nn.Linear(256, n_actions)

    def forward(self, obs):
        """
        Propagación hacia adelante:
        1) Convoluciones + ReLU (extracción espacial)
        2) Aplanamiento
        3) Fully-connected + ReLU (representación no lineal)
        4) Capa de salida (Q-values)
        """
        # 1) Extracción de características espaciales
        x = F.relu(self.conv1(obs))  # Reduce dimensionalidad y filtra información relevante
        x = F.relu(self.conv2(x))    # Afina patrones para la estimación de Q-values

        # 2) Aplanamiento de mapas de características a vector
        # Por qué: prepara la entrada para capas densas que no manejan tensores 2D
        x = x.view(x.size(0), -1)

        # 3) Transformación no lineal en espacio de características
        x = F.relu(self.fc(x))

        # 4) Capa final sin activación
        q_values = self.out(x)       # Devuelve un Q-value por acción, sin escalamiento
        return q_values