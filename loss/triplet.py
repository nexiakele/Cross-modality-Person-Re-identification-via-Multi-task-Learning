import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat1, feat2, label1, label2):
		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		#loss = Variable(.cuda())
		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, self.dist(center1, center2) - self.margin)
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center2) - self.margin)
				else:
					dist += max(0, 1-self.dist(center1, center2) - self.margin)

		return dist
    
class Cross_modal_TripletLoss3(nn.Module):
    def __init__(self, margin=0):
        super(Cross_modal_TripletLoss3, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, modal1_inputs, modal2_inputs, targets):
        n = modal1_inputs.size(0)
        r_dist  = self.compute_dist(modal1_inputs, modal1_inputs)
        t_dist  = self.compute_dist(modal2_inputs, modal2_inputs)
        rt_dist = self.compute_dist(modal1_inputs, modal2_inputs)
        tr_dist = self.compute_dist(modal2_inputs, modal1_inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap1, dist_an1 = self.get_hardest(r_dist, mask, n)
        dist_ap2, dist_an2 = self.get_hardest(t_dist, mask, n)
        dist_ap3, dist_an3 = self.get_hardest(rt_dist, mask, n)
        dist_ap4, dist_an4 = self.get_hardest(tr_dist, mask, n)
        
        dist_ap5, dist_an5 = dist_ap3[:], dist_an1[:]
        dist_ap6, dist_an6 = dist_ap4[:], dist_an2[:]
        
        
        dist_ap = torch.tensor(dist_ap1+dist_ap2+dist_ap3+dist_ap4+dist_ap5+dist_ap6).cuda()
        dist_an = torch.tensor(dist_an1+dist_an2+dist_an3+dist_an4+dist_an5+dist_an6).cuda()
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum().float() / y.size(0)
        return loss, prec
    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, inputs1, inputs2.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    def get_hardest(self, dist, mask, n):
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        return dist_ap, dist_an

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive

        
class CenterLoss(nn.Module):
    """
    paper: http://ydwen.github.io/papers/WenECCV16.pdf
    code:  https://github.com/pangyupo/mxnet_center_loss
    pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    """

    def __init__(self, features_dim, num_class=10, lamda=1., scale=1.0, batch_size=64):
        """
        初始化
        :param features_dim: 特征维度 = c*h*w
        :param num_class: 类别数量
        :param lamda   centerloss的权重系数 [0,1]
        :param scale:  center 的梯度缩放因子
        :param batch_size:  批次大小
        """
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        # store the center of each class , should be ( num_class, features_dim)
        self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))
        # self.lossfunc = CenterLossFunc.apply

    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: conv层输出的特征,  [b,c,h,w]
        :param y_truth:  标签值  [b,]
        :return:
        """
        batch_size = y_truth.size(0)
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim
        factor = self.scale / batch_size
        # return self.lamda * factor * self.lossfunc(output_features, y_truth, self.feature_centers))

        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_batch
        loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())
        #########
        return loss

class Cross_modal_ContrastiveLoss6(nn.Module):
    def __init__(self, margin=0.5):
        super(Cross_modal_ContrastiveLoss6, self).__init__()
        self.margin = margin
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            center_feat = (feat1.mean(dim=0) + feat2.mean(dim=0)) / 2.0
            centers.append(center_feat.unsqueeze(0))
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
              
        ################################        
        centers = torch.cat(centers, 0).cuda()
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        
        
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist1 = self.compute_dist(centersR,centers)
        dist2 = self.compute_dist(centersT,centers)
        dd1   = torch.sqrt(dist1 + 1e-10)
        dd2   = torch.sqrt(dist2 + 1e-10)
        
        label  = mask.float()
        loss1 = self.compute_loss(dist1, dd1, label)
        loss2 = self.compute_loss(dist2, dd2, label)
        return loss1 + loss2, 0

    def compute_loss(self, d2, d1, label):
        loss_contrastive = torch.mean((label) * torch.pow(d2,2) + (1.0-label) * torch.pow(torch.clamp(self.margin - d1, min=0.0), 2)) 
        return loss_contrastive

    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, inputs1, inputs2.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


class Cross_modal_ContrastiveLoss7(nn.Module):
    def __init__(self, margin=0.5):
        super(Cross_modal_ContrastiveLoss7, self).__init__()
        self.margin = margin
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            center_feat = (feat1.mean(dim=0) + feat2.mean(dim=0)) / 2.0
            centers.append(center_feat.unsqueeze(0))
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
              
        ################################        
        centers = torch.cat(centers, 0).cuda()
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        
        
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist1 = self.compute_dist(centersR,centers)
        dist2 = self.compute_dist(centersT,centers)
        
        label  = mask.float()
        loss1 = self.compute_loss(dist1, dist1, label)
        loss2 = self.compute_loss(dist2, dist2, label)
        return loss1 + loss2, 0

    def compute_loss(self, d2, d1, label):
        loss_contrastive = torch.mean((label) * torch.pow(d2,2) + (1.0-label) * torch.pow(torch.clamp(self.margin - d1, min=0.0), 2)) 
        return loss_contrastive

    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, inputs1, inputs2.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist    

class CenterLoss1(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=395, feat_dim=512, use_gpu=True):
        super(CenterLoss1, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
class Cross_modal_Center_ContrastiveLoss(nn.Module):
    def __init__(self, num_classes=395, feat_dim=512, margin=0.0):
        super(Cross_modal_Center_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim).cuda())
        self.mse_loss = nn.SmoothL1Loss()
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
        ################################################################        
        centers = self.centers.index_select(0, targets.long())
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        ################################################################
        loss1 = self.mse_loss(centersR, centers)
        loss2 = self.mse_loss(centersT, centers)
        return loss1 + loss2, 0


class Cross_modal_Center_ContrastiveLoss2(nn.Module):
    def __init__(self, num_classes=395, feat_dim=512, margin=0.5):
        super(Cross_modal_Center_ContrastiveLoss2, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim).cuda())
        self.mse_loss = nn.SmoothL1Loss()
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
        ################################################################        
        centers = self.centers.index_select(0, targets.long())
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        ################################################################
        loss1 = self.mse_loss(centersR, centers)
        loss2 = self.mse_loss(centersT, centers)
        ################################################################
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist = self.compute_dist(centers,centers)
        label  = mask.float()
        loss3 = torch.mean((1.0-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)) 
#        print(loss1.item(), loss2.item(), loss3.item(),  dist.max().item(),  dist.min().item(), dist.mean().item())
        return loss1 + loss2 + loss3, 0

    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, inputs1, inputs2.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
class Cross_modal_Center_ContrastiveLoss3(nn.Module):
    def __init__(self, num_classes=395, feat_dim=512, margin=0.5):
        super(Cross_modal_Center_ContrastiveLoss3, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim).cuda())
        self.mse_loss = nn.SmoothL1Loss()
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
        ################################################################        
        centers = self.centers.index_select(0, targets.long())
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        ################################################################
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        ################################################################
        dist1 = self.compute_dist(centersR,centers)
        dist2 = self.compute_dist(centersT,centers)
        ################################################################
        label  = mask.float()
        loss1 = self.compute_loss(dist1, dist1, label)
        loss2 = self.compute_loss(dist2, dist2, label)

        return loss1 + loss2, 0

    def compute_loss(self, d2, d1, label):
        loss_contrastive = torch.mean((label) * torch.pow(d2,2) + (1.0-label) * torch.pow(torch.clamp(self.margin - d1, min=0.0), 2)) 
        return loss_contrastive

    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()

        dist.addmm_(1, -2, inputs1, inputs2.t())

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist   
    
class Cross_modal_Center_ContrastiveLoss5(nn.Module):
    def __init__(self, num_classes=395, feat_dim=512, margin=0.0):
        super(Cross_modal_Center_ContrastiveLoss5, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim).cuda())
        self.mse_loss = nn.MSELoss()
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []
        ################################################################
        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
        ################################################################        
        centers = self.centers.index_select(0, targets.long())
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        ################################################################
        loss1 = self.mse_loss(centersR, centers)
        loss2 = self.mse_loss(centersT, centers)
        loss3 = self.mse_loss(centersR, centersT)
        return (loss1 + loss2), loss3, 0

if __name__ == '__main__':
#    tri = Cross_modal_TripletLoss3(0.75)
#    a = torch.tensor([[0,1,1], [0,2,0], [1,2,0], [1,4,1]]).float()
#    l = torch.tensor([1,1,2,2])
#    b = torch.tensor([[0,1,3], [0,1,0], [3,1,5], [1,2,5]]).float()
##    ab = torch.tensor([[0,1,3], [0,1,0], [3,1,5], [1,2,5], [0,1,1], [0,2,0], [1,2,0], [1,4,1]]).float()
##    ls = torch.tensor([1,2,3,4,1,2,3,4])
#
##    tri.forward( ab, ls)  
#    loss, prec=tri.forward(a, b, l) 
##    print(b.shape)
#    print(loss, prec)
##    l1 = [1,2,3]
#    l2 = [3,4,5]
#    print(l1+l2)
    num = [1,2,3,4,5,6,7,8,9]
