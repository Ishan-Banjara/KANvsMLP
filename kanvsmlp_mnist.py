import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import csv

class KANLayer(nn.Module):
  def __init__(self,
               input_features,
               output_features,
               grid_size=5,
               spline_order=3,
               grid_range=[-1,1],
               base_activation=nn.SiLU()):
    super().__init__()
    self.grid_range = grid_range
    self.spline_order = spline_order
    self.input_features = input_features # Assign input_features to self
    self.output_features = output_features # Assign output_features to self
    h = (self.grid_range[1]-self.grid_range[0])/grid_size
    #Spline of order k (degree k-1) with n+1 control points requires k+n+1 knots
    self.grid = torch.linspace(
        grid_range[0]-h*spline_order,
        grid_range[1]+h*spline_order,
        grid_size + 2*spline_order + 1
    ).expand(input_features,-1).contiguous()

    self.wb = nn.Parameter(torch.ones((output_features, input_features)))
    nn.init.xavier_uniform_(self.wb)
    self.ws = nn.Parameter(torch.ones((output_features, input_features, grid_size + spline_order)))
    self.base_activation = base_activation
    self.layernorm = nn.LayerNorm(output_features)   #Apply on d

  def forward(self, x:torch.Tensor):
    grid = self.grid.to(x.device)
    #Get base output = wb*b
    base_output = F.linear(self.base_activation(x), self.wb)

    #spline(x)
    x_unsqueeze = x.unsqueeze(-1)
    # Ensure bases has the correct dimensions for the linear operation later
    bases = ((x_unsqueeze >= grid[:, :-1]) & (x_unsqueeze <= grid[:, 1:])).to(x.dtype).to(x.device)


    for k in range(1, self.spline_order + 1):
      left_intervals = grid[:,:-(k+1)]
      right_intervals = grid[:,k:-1]
      bases = ((x_unsqueeze-left_intervals)/(right_intervals-left_intervals + 1e-9) * bases[:,:,:-1]) + \
               ((grid[:,k+1:]-x_unsqueeze)/(grid[:,k+1:]-grid[:,1:(-k)] + 1e-9)*bases[:,:,1:])
    bases = bases.contiguous()

    spline_out = F.linear(bases.view(x.size(0), -1), self.ws.view(self.ws.size(0), -1))
    return self.layernorm(base_output + spline_out) # Apply layernorm to the sum of base and spline outputs

class KAN(nn.Module):
  def __init__(self, input_features, output_features, hidden_layers):
    super().__init__()
    self.input_features = input_features

    self.layers = nn.ModuleList()
    in_size = input_features

    for out_size in hidden_layers:
      self.layers.append(KANLayer(in_size, out_size))
      in_size = out_size

    self.layers.append(KANLayer(in_size, output_features))

  def forward(self, x):
    x = x.view(-1, self.input_features)   #Flatten it to input
    for layer in self.layers:
        x = layer(x)
    return x

class MLP(nn.Module):
  def __init__(self, input_features, output_features, hidden_layers):
    super().__init__()
    self.input_features = input_features

    in_size = input_features
    self.layers = nn.ModuleList()
    for out_size in hidden_layers:
      self.layers.append(nn.Linear(in_size, out_size))
      self.layers.append(nn.ReLU())
      in_size = out_size
    self.layers.append(nn.Linear(in_size, output_features))

  def forward(self, x):
    x = x.view(-1, self.input_features)
    for layer in self.layers:
        x = layer(x)
    return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size = 256, shuffle=False)

input_features = 784
output_features = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Hyperparameters grid
param_grid = {
    'hidden_layers': [[512],
                      [512, 256],
                      [512, 256, 128],
                      [512, 16]],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
    'batch_size': [32, 64]
}

####CONFIGURATIONS###
k_folds = 5
num_epochs = 20

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)


#For KAN
results = {}
param_counts = {}
print("KAN model")
print("="*30)
for hidden_config in param_grid['hidden_layers']:
  for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
      config_str = f"hidden layers:{hidden_config}\nbatch size:{bs}\nlearning rate:{lr}"
      print(f"For configuration: \n{config_str}")
      print("="*30)
      fold_accuracies = []
      for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset)):
        print(f"Fold {fold+1}/{k_folds}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(trainset, batch_size=bs, sampler=train_subsampler)
        val_loader = DataLoader(trainset, batch_size=bs, sampler=val_subsampler)

        #Model initialization
        kan = KAN(input_features=input_features, output_features=output_features, hidden_layers = hidden_config).to(device)
        #We need this count for only once
        if fold == 0:
          num_params = sum(p.numel() for p in kan.parameters() if p.requires_grad)
          param_counts[config_str] = num_params
        optimizer = optim.Adam(kan.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        kan.train()
        for epoch in range(num_epochs):
          for data, targets in train_loader:
              data, targets = data.to(device), targets.to(device)
              optimizer.zero_grad()
              output = kan(data)
              loss = criterion(output, targets)
              loss.backward()
              optimizer.step()

        kan.eval()
        correct, total = 0, 0
        with torch.no_grad():
          for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            output = kan(data)
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100. * correct/total
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.2f}%")

      avg_accuracy = sum(fold_accuracies)/len(fold_accuracies)
      results[config_str] = avg_accuracy
      print(f" Average accuracy for this config: {avg_accuracy:.2f}%\n")

headers = ['hidden_layers', 'batch_size', 'learning_rate', '#parameters', 'average_accuracy']
with open('kan_results.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(headers)

  for config_str, avg_accuracy in results.items():
    lines = config_str.strip().split('\n')
    hidden_config = lines[0].split(':')[1]
    batch_size = lines[1].split(':')[1]
    learning_rate = lines[2].split(':')[1]
    num_params = param_counts[config_str]
    writer.writerow([hidden_config, batch_size, learning_rate, num_params, avg_accuracy])


best_config_str = max(results, key=results.get)
best_accuracy = results[best_config_str]
print(f"Finished Tuning!!!")
print(f"Best Configuration for KAN: {best_config_str}")
print(f"Best Average K-Fold Accuracy for KAN: {best_accuracy:.2f}%")

#For MLP
results = {}
param_counts = {}
print("MLP model")
print("="*30)

for hidden_config in param_grid['hidden_layers']:
  for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
      config_str = f"hidden layers:{hidden_config}\nbatch size:{bs}\nlearning rate:{lr}"
      print(f"For configuration: \n{config_str}")
      print("="*30)
      fold_accuracies = []
      for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset)):
        print(f"Fold {fold+1}/{k_folds}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(trainset, batch_size=bs, sampler=train_subsampler)
        val_loader = DataLoader(trainset, batch_size=bs, sampler=val_subsampler)

        #Model initialization
        mlp = MLP(input_features=input_features, output_features=output_features, hidden_layers = hidden_config).to(device)
        if fold==0:
          num_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
          param_counts[config_str] = num_params
        optimizer = optim.Adam(mlp.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        mlp.train()
        for epoch in range(num_epochs):
          for data, targets in train_loader:
              data, targets = data.to(device), targets.to(device)
              optimizer.zero_grad()
              output = mlp(data)
              loss = criterion(output, targets)
              loss.backward()
              optimizer.step()

        mlp.eval()
        correct, total = 0, 0
        with torch.no_grad():
          for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            output = mlp(data)
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100. * correct/total
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.2f}%")

      avg_accuracy = sum(fold_accuracies)/len(fold_accuracies)
      results[config_str] = avg_accuracy
      print(f" Average accuracy for this config: {avg_accuracy:.2f}%\n")

headers = ['hidden_layers', 'batch_size', 'learning_rate', '#parameters', 'average_accuracy']
with open('mlp_results.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(headers)

  for config_str, avg_accuracy in results.items():
      lines = config_str.strip().split('\n')
      hidden_config = lines[0].split(':')[1]
      batch_size = lines[1].split(':')[1]
      learning_rate = lines[2].split(':')[1]
      num_params = param_counts[config_str]
      writer.writerow(
        [hidden_config, batch_size, learning_rate, num_params,
         avg_accuracy])

best_config_str = max(results, key=results.get)
best_accuracy = results[best_config_str]
print(f"Finished Tuning!!!")
print(f"Best Configuration for MLP: {best_config_str}")
print(f"Best Average K-Fold Accuracy for MLP: {best_accuracy:.2f}%")