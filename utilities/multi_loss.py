import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(1), requires_grad=True)

    def forward(self, loss1, loss2):
       alpha = torch.sigmoid(self.alpha).cuda()
       loss = alpha*loss1+(1-alpha)*loss2
       return loss