import torch
# from torchvision.models import efficientnet_b0
from blockselect import block_selector
import Lenet5
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchbearer import Trial
import numpy as np
from torchvision.transforms import Compose, Normalize
from torchbearer import Trial
import torchbearer
import torch.nn as nn

from FMix.models import ResNet18



model = ResNet18(nc=1)
model_file = "./base_models/model_1_FMIX_fashion.pt"
if(torch.cuda.is_available()):
  state = torch.load(model_file)
else:
  state = torch.load(model_file, map_location="cpu")
model.load_state_dict(state["model"])
normalise = Normalize((0.1307,), (0.3081,))
base = [ToTensor(), normalise]
transform_test = Compose(base)
test_ds = FashionMNIST("fashdata", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
test_loader_2 = DataLoader(test_ds, batch_size=1)


# t = Trial(model,criterion=nn.CrossEntropyLoss(), metrics=["acc", 'loss', 'lr'])
# t.load_state_dict(state)
# t.with_test_generator(test_loader)
# t.with_val_generator(test_loader)
# model.load_state_dict(t.state_dict()["model"])
# x = t.for_steps(val_steps=1, test_steps=1).predict(data_key=torchbearer.TEST_DATA)
model.eval()
# print(x)

true = 0
total = 0
labels = []

for input,out in test_loader_2:
  total += 1
  res = model(input)
  if(res.argmax() == out[0]):
    true += 1
  labels.append(res.argmax())
  print(f"Acc {true/total}\n",end="")

# xmax = x.argmax(1)


# t.with_test_generator(test_loader)
# test_predictions = t.evaluate(data_key=torchbearer.TEST_DATA)
# loss = torch.nn.CrossEntropyLoss()
# my_block_calc = block_selector.block_entropy_calc(model, train_loader, loss)
# result = my_block_calc.forward()

x = np.array([])
# print(result)
# np.save("./block_npz", result)




