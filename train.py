import torch
from utils.categories import image_net_categories
from utils.getDevice import getDevice
from matplotlib import pyplot as plt
from AlexNet import AlexNet
from torch import nn
from torchvision import transforms as torchTrans
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchsummary import summary as torchsummary

from torchvision import datasets

# get Fashion MNist
def get_datas (batch_size):
  trans = torchTrans.Compose([
    torchTrans.Resize((224, 224)),
    torchTrans.ToTensor()
  ])

  training_data = datasets.FashionMNIST(
    root="Fashion_MNIST_Data",
    train=True,
    download=True,
    transform=trans,
  )
  test_data = datasets.FashionMNIST(
    root="Fashion_MNIST_Data",
    train=False,
    download=True,
    transform=trans,
  )

  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)

  def get_label (idx):
    return training_data.classes[idx]

  return training_data, test_data, train_dataloader, test_dataloader, get_label

# get_datas for ImageNet
# def get_datas (batch_size):
#   trans = torchTrans.Compose([
#     torchTrans.Resize((224, 224)),
#     torchTrans.ToTensor()
#   ])

#   datas = ImageFolder(
#     root = "ImageNet_DATA/val",
#     transform = trans
#   )

#   train_dataloader = DataLoader(
#     dataset = datas,
#     batch_size = batch_size,
#     shuffle = True,
#   )

#   test_dataloader = DataLoader(
#     dataset = datas,
#     batch_size = batch_size,
#   )

#   def get_label (idx):
#     return image_net_categories[datas.classes[idx]]

#   return datas, datas, train_dataloader, test_dataloader, get_label

def print_imgs (train_dataloader, get_label):
  for _, (X, y) in enumerate(train_dataloader):
    for (X_i, y_i) in zip(X, y) :
      img = X_i.numpy().transpose(1, 2, 0)
      plt.title(get_label(y_i.item()))
      plt.imshow(img)
      plt.show()

def train (dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)

  # epoch
  for batch, (X, y) in enumerate(dataloader) :
    X, y = X.to(device), y.to(device)

    # forward propagation
    pred = model(X)
    loss = loss_fn(pred, y)

    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss every 10th learning
    if batch % 10 == 0:
      current = (batch + 1) * len(X)
      print(f"loss: {loss:>7f} [{current}/{size}]")

def test (dataloader, model, loss_fn):
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

if __name__ == '__main__':
  batch_size = 128
  epochs = 5
  learning_rate = 1e-3

  device = getDevice()
  print(f"Using {device} device")

  _, _, train_dataloader, test_dataloader, get_label = get_datas(batch_size)

  print_imgs(train_dataloader, get_label)

  model = AlexNet().to(device)
  torchsummary(
    model = model,
    input_size = (1, 224, 224),
    device = device
  )

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
  )

  for t in range(epochs):
    print(f"Epoch {t+1}\n{'-' * 50}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
  print("Done!")

  torch.save(model.state_dict(), "model.pth")
  print("Saved PyTorch Model State to model.pth")
