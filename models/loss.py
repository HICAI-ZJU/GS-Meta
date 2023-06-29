import torch
from torch import nn
import torch.nn.functional as F


class NCESoftmaxLoss(nn.Module):
    def __init__(self, t=0.08):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.t = t

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        emb = F.normalize(torch.cat([z_i, z_j]))
        similarity = torch.matmul(emb, emb.t()) - torch.eye(batch_size * 2).to(z_i.device) * 1e12
        similarity = similarity / self.t
        label = torch.tensor([(batch_size + i) % (batch_size * 2) for i in range(batch_size * 2)]).to(
            similarity.device).long()
        loss = self.criterion(similarity, label)
        return loss


if __name__ == '__main__':
    x, y = torch.randn(5, 300), torch.randn(5, 300)
    cri = NCESoftmaxLoss(t=0.08)
    loss = cri(x, y)
    print(loss)
