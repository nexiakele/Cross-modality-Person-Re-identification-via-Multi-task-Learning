import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.tensor(dist_ap)
        dist_an = torch.tensor(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class Cross_modal_TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(Cross_modal_TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, modal1_inputs, modal2_inputs, targets):
        n = modal1_inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(modal1_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(modal2_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, modal1_inputs, modal2_inputs.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        
        dist_ap = torch.tensor(dist_ap)
        dist_an = torch.tensor(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum().float() / y.size(0)
        return loss, prec

class totall_loss1(nn.Module):
    def __init__(self, margin=0, weight=[0.2, 1]):
        super(totall_loss1, self).__init__()
        self.margin = margin
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.triplet_loss = Cross_modal_TripletLoss()
        self.weight = weight
    def forward(self, pred_id, feat, labels):
        loss1 = self.cross_entropy_loss(pred_id, labels) * self.weight[0]
        b,c= feat.shape
        rgb_out = feat[:b//2,:]
        t_out   = feat[b//2:,:]
        loss2, prec2 = self.triplet_loss(rgb_out, t_out, labels[:b//2,:])
        loss3, prec3 = self.triplet_loss(t_out, rgb_out, labels[:b//2,:])
        return loss1 + loss2 + loss3
def get_loss(loss_type=0):
    if loss_type==0:
        loss = totall_loss1(0.5, [0.2, 1])
    return loss




    
if __name__ == '__main__':
    tri = Cross_modal_TripletLoss(0.5)
    a = torch.tensor([[1,1,1], [0,2,0], [1,2,1], [1,4,1]]).float()
    l = torch.tensor([1,2,3,4])
    b = torch.tensor([[2,1,3], [0,1,0], [4,1,8], [1,2,5]]).float()
    loss, prec = tri.forward(a, b, l)  
    print(loss, prec)