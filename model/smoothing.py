import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGaussianSmoothing(nn.Module):
    def __init__(self, kernel_size, sigma_init=1.0):
        super(LearnableGaussianSmoothing, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([sigma_init]))
        self.kernel = nn.Parameter(self._gaussian_kernel(kernel_size, self.sigma).unsqueeze(0).unsqueeze(0).expand(3,-1,-1))
        self.sigma.requires_grad = True

    def forward(self, x):
        # Apply 1D convolution with the learned Gaussian kernel
        #print(self.kernel.shape)
        if(len(x.shape)!=3):
            x = x.unsqueeze(0)
        smoothed = F.conv1d(x.transpose(1,2), self.kernel, groups=3, padding='same')
        return smoothed.transpose(1,2)

    def _gaussian_kernel(self, size, sigma):
        #interval = (2 * sigma**2) / (size - 1)
        x = torch.arange(size, dtype=torch.float32)
        kernel = torch.exp(-((x - size // 2) ** 2) / (2 * sigma ** 2))
        return kernel / torch.sum(kernel)
    


