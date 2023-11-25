"""
Aditya Patel
Implementation of the GAN Algorithm on the Gaussian Mixed Model
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from torch.autograd.variable import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# Bivariate Normal Distributions
def generate_gmm_samples(num_samples, means, std_dev):
  num_comps = len(means)
  weights = np.ones(num_comps) / num_comps
  samples = np.zeros((num_samples, 2))

  for i in range(num_samples):
    comp_idx = np.random.choice(num_comps, p=weights)
    mean = means[comp_idx]
    samples[i] = np.random.normal(loc=mean, scale=std_dev)
  
  return samples

# GAN Architecture
class Generator(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, output_size)
    )
  
  def forward(self, x):
    return self.model(x)

class Discriminator(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)

# Function to get GAN outputs from input  
def GAN_generator(gmm_samples, input_size = 100, output_size = 2, num_samples = 1600, bts=64, epoch_ct=10000, lr=0.001):
  # Generate Training data
  train_data = torch.zeros((num_samples, 2))
  train_data[:, 0] = torch.Tensor(gmm_samples[:, 0])
  train_data[:, 1] = torch.Tensor(gmm_samples[:, 1])
  train_labels = torch.zeros((num_samples))
  train_set = [
     (train_data[i], train_labels[i]) for i in range(num_samples)
  ]

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=bts, shuffle=True)
  
  # Initiatlization of components
  genr = Generator(2, output_size)
  disc = Discriminator(output_size)
  criterion = nn.BCELoss()
  optim_G = optim.Adam(genr.parameters(), lr=lr)
  optim_D = optim.Adam(disc.parameters(), lr=lr)

  # Training loop
  for epoch in range(epoch_ct):
    for n, (real_samples, _) in enumerate(train_loader):
        # Train Discriminator
        real_sample_labels = torch.mul(torch.ones((bts, 1)), 1)
        latent_space_samples = torch.randn((bts, 2))
       
        genned_samples = genr(latent_space_samples)
        genned_sample_labels = torch.zeros((bts, 1))

        all_samples = torch.cat((real_samples, genned_samples))
        all_sample_labels = torch.cat((real_sample_labels, genned_sample_labels))

        disc.zero_grad()
        disc_out = disc(all_samples)
        loss_disc = criterion(disc_out, all_sample_labels)
        loss_disc.backward()
        optim_D.step()

        # Train Generator
        latent_space_samples = torch.randn((bts, 2))

        genr.zero_grad()
        genned_samples = genr(latent_space_samples)
        disc_gen_out = disc(genned_samples)
        loss_gen = criterion(disc_gen_out, real_sample_labels)
        loss_gen.backward()
        optim_G.step()

    # Print loss
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epoch_ct}]: D Loss: {loss_disc:.4f} | G Loss: {loss_gen:.4f}")

  test = Variable(torch.randn(num_samples, input_size))
  genr_samples = genr(test)
  
  return genr_samples.detach().numpy()

# MMD function from provided reference
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":  
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)

def driver(num_components=8, samples=1600, stdev=0.02, input_size=100, lr=0.0002, bts=64, epoch_ct=10000):
  
  gmm_means = np.array([(2 * np.cos(2 * np.pi * k / num_components), 2 * np.sin(2 * np.pi * k / num_components)) for k in range(num_components)])

  gmm_samples = generate_gmm_samples(samples, gmm_means, stdev)
  gan_samples = GAN_generator(gmm_samples, input_size, num_samples=samples, bts=bts, epoch_ct=epoch_ct, lr=lr)

  # Create tensors and run MMD
  gan_tensor = torch.from_numpy(gan_samples)
  biv_tensor = torch.from_numpy(gmm_samples).to(dtype=torch.float32)
  mmd_score = MMD(gan_tensor, biv_tensor, kernel='multiscale')
  print(f"The MMD score was identified as {mmd_score:.4f}")

  # Plot the results
  plt.scatter(gan_samples[:, 0], gan_samples[:, 1], label='GAN Samples', alpha=0.5)
  plt.scatter(gmm_samples[:, 0], gmm_samples[:, 1], label='Mixture Samples', alpha=0.5)
  plt.title(f"Samples from GAN and Mixture of 8 Gaussians. \n MMD Score = {mmd_score:.4f}")
  plt.legend()
  plt.show()

  