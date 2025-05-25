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
        super(DQN_CNN_Model, self).__init__()
        in_channels, h, w = obs_shape

        # Capas convolucionales (igual que en el paper)
        print(in_channels, h, w)
        out_channels_conv1 = 16
        # self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=9, padding=4)
        kernel_size = 8
        stride = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels_conv1, kernel_size=kernel_size, stride=stride)
        # print(f"{ states.shape = }")
        h, w = conv2d_output_shape((h, w), kernel_size, stride=4)
        print('conv1', h, w)

        kernel_size = 4
        stride = 2
        self.conv2 = nn.Conv2d(out_channels_conv1, 32, kernel_size=kernel_size, stride=stride)
        h, w = conv2d_output_shape((h, w), kernel_size, stride=stride)
        print('conv2', h, w)

        # Capas completamente conectadas
        # TODO porque 256
        self.fc1 = nn.Linear(32 * h * w, 256)
        self.fc2 = nn.Linear(256, n_actions)
        print(self.fc2)
        

    def forward(self, obs):
        # TODO: 1) aplicar convoluciones y activaciones
        #       2) aplanar la salida
        #       3) aplicar capas lineales
        #       4) devolver tensor de Q-values de tamaño (batch, n_actions)

        # obs shape: (batch_size, 4, 84, 84)
        # print(obs)
        print(f"{ obs.shape = }")
        
        x = self.conv1(obs)
        x = F.relu(x)
        print(f"{ x.shape = }")

        x = self.conv2(x)
        print(f"{ x.shape = }")
        # result shape
        x = F.relu(x)
        print(f"{ x.shape = }")

        x = x.view(x.size(0), -1)  # Aplanar
        print(f"{ x.shape = }")
        x = F.relu(self.fc1(x))
        print(f"{ x.shape = }")
        return self.fc2(x)