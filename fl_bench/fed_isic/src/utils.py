import torch
from dotenv import load_dotenv
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

load_dotenv()


class WeightedFocalLoss(_Loss):

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
