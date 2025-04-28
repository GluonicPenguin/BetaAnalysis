import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SignalProbabilityModel(nn.Module):
  def __init__(self, ansatz_pmax_value, ansatz_tmax_value):
    super().__init__()
    self.ansatz_pmax_value = ansatz_pmax_value
    self.amp_mu = nn.Parameter(torch.tensor(2.0*ansatz_pmax_value))   # Langaus centre
    self.amp_sigma = nn.Parameter(torch.tensor(3.0)) # Langaus width
    self.pow_mu = nn.Parameter(torch.tensor(2.0*ansatz_pmax_value))  # Langaus centre
    self.pow_sigma = nn.Parameter(torch.tensor(3.0)) # Langaus width
    self.tmax_mu = nn.Parameter(torch.tensor(ansatz_tmax_value)) # Gaussian centre
    self.tmax_sigma = nn.Parameter(torch.tensor(0.5)) # Gaussian width

  def forward(self, amplitudes, pow, tmax):
    amp_prob = torch.sigmoid((amplitudes - self.amp_mu) / self.amp_sigma)
    pow_prob = torch.sigmoid((pow - self.pow_mu) / self.pow_sigma)
    tmax_prob = torch.sigmoid((tmax - self.tmax_mu) / self.tmax_sigma)
    return amp_prob * pow_prob * tmax_prob

def differential_programming_SigProbModel(num_epochs, pmax, width, tmax, ansatz_pmax_value, ansatz_tmax_value, learning_rate):
  model = SignalProbabilityModel(ansatz_pmax_value, ansatz_tmax_value)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  num_events = len(pmax)
  sig_events = np.count_nonzero(pmax >= ansatz_pmax_value)
  labels = np.concatenate([np.ones(sig_events), np.zeros(num_events - sig_events)])  # 1 = signal, 0 = noise

  # Convert to PyTorch tensors
  max_amplitudes = torch.tensor(pmax, dtype=torch.float32)
  max_pow = torch.tensor(pmax / width, dtype=torch.float32)
  max_tmax = torch.tensor(tmax, dtype=torch.float32)
  labels = torch.tensor(labels, dtype=torch.float32)

  for epoch in range(num_epochs):
    optimizer.zero_grad()
    scores = model(max_amplitudes, max_pow, max_tmax)
    loss = -torch.mean(labels * torch.log(scores + 1e-6) + (1 - labels) * torch.log(1 - scores + 1e-6))  
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item()}")
      #for name, param in model.named_parameters():
      #  print(name, param.grad)

  return model(max_amplitudes, max_pow, max_tmax).detach().numpy(), max_amplitudes
