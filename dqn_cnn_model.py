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
        # obs_shape = (C, H, W), típicamente C=4 (frames stack), H=W=84
        c, h, w = obs_shape

        # Capas convolucionales según el paper
        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculamos dinámicamente la salida de las convs para la capa fc
        h1, w1 = conv2d_output_shape((h, w), kernel_size=8, stride=4)
        print(f"conv1 output shape: {h1}x{w1}")
        h2, w2 = conv2d_output_shape((h1, w1), kernel_size=4, stride=2)
        print(f"conv2 output shape: {h2}x{w2}")
        conv_out_size = h2 * w2 * 32

        # Capa fully-connected intermedia
        self.fc = nn.Linear(conv_out_size, 256)
        # Capa de salida para Q-values
        self.out = nn.Linear(256, n_actions)
        print(f"conv_out_size: {conv_out_size}, fc output size: 256, n_actions: {n_actions}")
        print(self.out)
        

    def forward(self, obs):
        """
        Forward pass:
          1) ReLU(conv1)
          2) ReLU(conv2)
          3) Aplanar
          4) ReLU(fc)
          5) out Q-values
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        q_vals = self.out(x)
        return q_vals 