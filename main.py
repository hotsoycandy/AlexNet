import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.ImageNet(
  root = "ImageNet_DATA",
  split = "train",
  transform = ToTensor(),
)

test_data = datasets.ImageNet(
  root = "ImageNet_DATA",
  split = "val",
  transform = ToTensor(),
)

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

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
      torch.nn.LocalResponseNorm(
        size = 5,
        alpha = 1e-4,
        beta = 0.75,
        k = 2,
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
      torch.nn.LocalResponseNorm(
        size = 5,
        alpha = 1e-4,
        beta = 0.75,
        k = 2,
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

      # Layer4. Convolutional Layer
      nn.Conv2d(
        in_channels = 384,
        out_channels = 384,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = True,
      ),
      nn.ReLU(),

      # Layer5. Convolutional Layer
      nn.Conv2d(
        in_channels = 384,
        out_channels = 256,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        bias = True,
      ),
      nn.ReLU(),
      nn.MaxPool2d(
        kernel_size = 3,
        stride = 2,
      ),

      nn.Flatten(),

      # Layer6. Affine Layer
      nn.Linear(9216, 4096),
      nn.Dropout(p = 0.5),
      nn.ReLU(),

      # Layer7. Affine Layer
      nn.Linear(4096, 4096),
      nn.Dropout(p = 0.5),
      nn.ReLU(),

      # Layer8. Affine Layer
      nn.Linear(4096, 1000),
      nn.Softmax(1),
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
  model.parameters(),
  lr = 0.01,
  momentum = 0.9,
  weight_decay = 1e-4 * 5)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # 예측 오류 계산
    pred = model(X)
    loss = loss_fn(pred, y)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
