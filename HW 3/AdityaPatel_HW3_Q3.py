"""
Aditya Patel
Homework 3 Q3
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GaussianMonteCarlo(num_samples, num_components, means, var):
    covars = np.array([[[var, 0],[0, var]]] * num_components)
    weights = np.ones(comps) / comps
    gmm_mc_samples = np.zeros((num_samples, len(means[0])))

    for i in range(num_samples):
        # Sample component index
        comp_idx = np.random.choice(num_components, p=weights)

        # Sample from selected component using normal distribution pdf
        sample = np.random.multivariate_normal(means[comp_idx], covars[comp_idx])
        gmm_mc_samples[i] = sample
    
    return gmm_mc_samples

def BivariateNormSamples(num_samples, num_components, means, var):
    covars = np.array([[[var, 0],[0, var]]] * num_components)
    
    biv_norm_samples = np.concatenate([
            np.random.multivariate_normal(means[k], covars[k], size=int(num_samples/num_components)) for k in range(num_components)
    ])
    
    return biv_norm_samples

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

if __name__ == '__main__':
    # Define GMM parameters
    samps = 800
    comps = 8
    variance = 0.02*0.02
    gmm_means = np.array([(2 * np.cos((2 * np.pi * k)/comps), 2 * np.sin((2 * np.pi * k)/comps)) for k in range(comps)])
    

    # Obtain samples from Monte Carlo method
    gmm_mc_samples = GaussianMonteCarlo(samps, comps, gmm_means, variance)
    
    # Obtain samples from Bivariate Normal distribution
    biv_norm_samples = BivariateNormSamples(800, 8, gmm_means, variance)

    # Create tensors and run MMD
    gmm_tensor = torch.from_numpy(gmm_mc_samples)
    biv_tensor = torch.from_numpy(biv_norm_samples)
    mmd_score = MMD(gmm_tensor, biv_tensor, kernel='multiscale')

    print("The MMD score was identified as {}".format(mmd_score))

    # Plot the results
    plt.scatter(biv_norm_samples[:, 0], biv_norm_samples[:, 1], label='Individual Samples', alpha=0.5)
    plt.scatter(gmm_mc_samples[:, 0], gmm_mc_samples[:, 1], label='GMM Samples', alpha=.5)
    plt.title('Samples from GMM and Individual Bivariate Normals.\nMMD Score: {:.4f}'.format(mmd_score))
    plt.legend()
    plt.show()