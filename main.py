import torch
from torch import nn

# 모델을 정의합니다.
class NeuralNetwork (nn.Module) :
  def __init__ (self) :
    super().__init__()

    self.linear_relu_stack = nn.Sequential(
      # Layer1. Convolutional Layer
      nn.Conv2d(
        in_channels = 3,
        out_channels = 96,
        kernel_size = 11,
        stride = 4,
        padding = 2,
      ),
      nn.ReLU(),
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
        bias = True,
      ),
      nn.ReLU(),
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
        bias = True,
      ),
      nn.ReLU(),
    )

  def forward (self, x):
    logits = self.linear_relu_stack(x)
    return logits

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

input = torch.randn((100, 3, 224, 224)).to(device)
print(input.shape)

output = model.forward(input)
print(output.shape)
