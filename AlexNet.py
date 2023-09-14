from matplotlib import pyplot as plt

from torch import nn
from utils.getDevice import getDevice
from torchsummary import summary as torchsummary

# 모델을 정의합니다.
class AlexNet (nn.Module) :
  def __init__ (self) :
    super(AlexNet, self).__init__()

    self.layers = nn.Sequential(
      # Layer1. Convolutional Layer
      nn.Conv2d(
        in_channels = 3,
        out_channels = 96,
        kernel_size = 11,
        stride = 4,
        padding = 2,
      ),
      nn.ReLU(),
      nn.LocalResponseNorm(
        size = 5,
        alpha = 1e-4,
        beta = 0.75,
        k = 2,
      ),
      nn.MaxPool2d(
        kernel_size = 3,
        stride = 2,
      ),

      # Layer2. Convolutional Layer
      nn.Conv2d(
        in_channels = 96,
        out_channels = 256,
        kernel_size = 5,
        stride = 1,
        padding = 2,
      ),
      nn.ReLU(),
      nn.LocalResponseNorm(
        size = 5,
        alpha = 1e-4,
        beta = 0.75,
        k = 2,
      ),
      nn.MaxPool2d(
        kernel_size = 3,
        stride = 2,
      ),

      # Layer3. Convolutional Layer
      nn.Conv2d(
        in_channels = 256,
        out_channels = 384,
        kernel_size = 3,
        stride = 1,
        padding = 1,
      ),
      nn.ReLU(),

      # Layer4. Convolutional Layer
      nn.Conv2d(
        in_channels = 384,
        out_channels = 384,
        kernel_size = 3,
        stride = 1,
        padding = 1,
      ),
      nn.ReLU(),

      # Layer5. Convolutional Layer
      nn.Conv2d(
        in_channels = 384,
        out_channels = 256,
        kernel_size = 3,
        stride = 1,
        padding = 1,
      ),
      nn.ReLU(),
      nn.MaxPool2d(
        kernel_size = 3,
        stride = 2,
      ),

      nn.Flatten(),

      # Layer6. Affine Layer
      nn.Dropout(p = 0.5),
      nn.Linear(9216, 4096),
      nn.ReLU(),
      # Layer7. Affine Layer
      nn.Dropout(p = 0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      # Layer8. Affine Layer
      nn.Linear(4096, 1000),
    )

  def forward (self, x) :
    x = self.layers(x)
    return x

if __name__ == '__main__' :
  device = getDevice()
  model = AlexNet().to(device)
  torchsummary(
    model = model,
    input_size = (3, 224, 224),
    device = device
  )
