import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
############################################################################### 
############################################################################### 
def conv1x1xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
###############################################################################
############################################################################### 
def conv3x3xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
###############################################################################  
###############################################################################   
class basic_deconv_up(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel,kernel_size=3, stride = 2,
                                           padding = 1,output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
    def forward(self,x):
        out = self.deconv(x)
        return out
###############################################################################  
###############################################################################      
def get_resnet_layers1(pretrained = True):
    if pretrained:
        print('using pretrained weight!')
    features = list(models.resnet50(pretrained=pretrained).children())[0:5]
    conv1  = features[0]
    bn  = features[1]
    relu = features[2]
    maxpool = features[3]
    block1 = nn.Sequential(conv1, bn, relu, maxpool)
    block2 = features[4]
    return block1, block2
###############################################################################  
###############################################################################  
def get_resnet_layers2(pretrained = True):
    if pretrained:
        print('using pretrained weight!')
    features = list(models.resnet50(pretrained=pretrained).children())[5:8]
    block1 = features[0]
    block2 = features[1]
    block3 = features[2]
    for mo in block3.modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
    return block1, block2, block3
###############################################################################  
###############################################################################  
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out * 5
###############################################################################  
###############################################################################  
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
###############################################################################  
###############################################################################  

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
###############################################################################  
###############################################################################  

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
###############################################################################  
###############################################################################  

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       
###############################################################################  
###############################################################################  
    
def part(x, num_part):
    sx = x.size(2) / num_part
    sx = int(sx)
    kx = x.size(2) - sx * (num_part - 1)
    kx = int(kx)
    x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
    #x = self.visible.avgpool(x)
    x = x.view(x.size(0), x.size(1), x.size(2))
    return x
# Define the ResNet18-based Model
# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(visible_net_resnet, self).__init__()
        self.b1,self.b2 = get_resnet_layers1(pretrained=True)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x
###############################################################################  
###############################################################################  
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(thermal_net_resnet, self).__init__()
        self.b1,self.b2 = get_resnet_layers1(pretrained=True)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x
###############################################################################  
###############################################################################  
class Shared_net_resnet(nn.Module):
    def __init__(self, pool_dim = 2048, num_class=395, arch ='resnet18'):
        super(Shared_net_resnet, self).__init__()
        self.b1, self.b2, self.b3= get_resnet_layers2(pretrained=True)
        
        self.clss1 = nn.Conv2d(512, num_class, 3,2,1)
        self.dconv1 = nn.Sequential(conv3x3xbnxrelu(512, 256),
                                    conv3x3xbnxrelu(256, 256))
        
        self.dconv2 = nn.Sequential(basic_deconv_up(256, 128),
                                    conv3x3xbnxrelu(128, 128))
        
        self.pred = nn.Sequential(nn.Conv2d(128, 1, 3,1,1),
                                  nn.Sigmoid())
        
###############################################################################  
###############################################################################  
        self.conv1 =  conv3x3xbnxrelu(128, 256, 2)
        self.conv2 =  nn.Sequential(conv3x3xbnxrelu(256, 512, 2),
                                    conv1x1xbnxrelu(512, 2048))

###############################################################################  
###############################################################################  
        self.skip =  conv3x3xbnxrelu(256, 1024, 2)
###############################################################################  
###############################################################################  
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l2norm = Normalize(2)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self, x):
        b,c,h,w = x.shape
        x1 = self.b1(x)
        ids = self.clss1(x1)
        ids = self.avgpool(ids).view(ids.size(0), -1)
        #########salient map 预测 ###################################
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        maps = self.pred(dx2)
        ########################################################################
        ########################################################################
        tx1 = self.conv1(dx2)
        fusion = self.conv2(tx1 + dx1)
        skip = self.skip(tx1)
        ########################################################################
        #########融合saliency 特征和ReID 特征 ###################################
        ##########融合Re-ID的特征提取 ###################################
        x2 = self.b2(x1)
        x2 = skip + x2
        x3 = self.b3(x2)
        x3 = fusion + x3
        ##########Re-ID的特征提取 ###################################
        sx26 = part(x3, 6)
        sx26 = sx26.chunk(6, 2)
        b = sx26[1].size(0)
        xs1 = sx26[0].contiguous().view(b,-1)
        xs2 = sx26[1].contiguous().view(b,-1)
        xs3 = sx26[2].contiguous().view(b,-1)
        xs4 = sx26[3].contiguous().view(b,-1)
        xs5 = sx26[4].contiguous().view(b,-1)
        xs6 = sx26[5].contiguous().view(b,-1)
#        ##########多层级特征提取 ###################################
        fu  = part(fusion, 6)
        ssx26 = fu.chunk(6, 2)
        b = ssx26[1].size(0)
        xxs1 = ssx26[0].contiguous().view(b,-1)
        xxs2 = ssx26[1].contiguous().view(b,-1)
        xxs3 = ssx26[2].contiguous().view(b,-1)
        xxs4 = ssx26[3].contiguous().view(b,-1)
        xxs5 = ssx26[4].contiguous().view(b,-1)
        xxs6 = ssx26[5].contiguous().view(b,-1)
        avg1 = self.avgpool(x2).view(x2.size(0),-1)
        avg2 = self.avgpool(x3).view(x3.size(0),-1)
        avg3 = self.avgpool(tx1).view(tx1.size(0),-1)
        avg4 = self.avgpool(fusion).view(fusion.size(0),-1)
        return  xs1, xs2, xs3, xs4, xs5, xs6,xxs1, xxs2, xxs3, xxs4, xxs5, xxs6, maps, ids, (avg1, avg2, avg3, avg4)

class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 2048
        low_dim=512
        self.shared = Shared_net_resnet(num_class=class_num)
        self.feature1 = FeatureBlock(2048, low_dim, dropout = drop)
        self.feature2 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature7 = FeatureBlock(2048, low_dim, dropout = drop)
        self.feature8 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature9 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature10 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature11 = FeatureBlock(2048, low_dim, dropout=drop)
        self.feature12 = FeatureBlock(2048, low_dim, dropout=drop)
        
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier7 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier8 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier9 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier10 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier11 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier12 = ClassBlock(low_dim, class_num, dropout=drop)
        
        self.l2norm = Normalize(2)
        
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            xx1 = self.visible_net(x1)
            xx2 = self.thermal_net(x2)
            xx = torch.cat((xx1, xx2),0)
        elif modal ==1:
            xx = self.visible_net(x1)
        elif modal ==2:
            xx  = self.thermal_net(x2)
            
        x_0, x_1, x_2, x_3, x_4,x_5,x_6, x_7, x_8, x_9, x_10,x_11, maps, ids, avgs  = self.shared(xx)
        
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        y_6 = self.feature7(x_6)
        y_7 = self.feature8(x_7)
        y_8 = self.feature9(x_8)
        y_9 = self.feature10(x_9)
        y_10 = self.feature11(x_10)
        y_11 = self.feature12(x_11)
        
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        out_6 = self.classifier7(y_6)
        out_7 = self.classifier8(y_7)
        out_8 = self.classifier9(y_8)
        out_9 = self.classifier10(y_9)
        out_10 = self.classifier11(y_10)
        out_11 = self.classifier12(y_11)
        
        y_0 = self.l2norm(y_0)
        y_1 = self.l2norm(y_1)
        y_2 = self.l2norm(y_2)
        y_3 = self.l2norm(y_3)
        y_4 = self.l2norm(y_4)
        y_5 = self.l2norm(y_5)
        y_6 = self.l2norm(y_6)
        y_7 = self.l2norm(y_7)
        y_8 = self.l2norm(y_8)
        y_9 = self.l2norm(y_9)
        y_10 = self.l2norm(y_10)
        y_11 = self.l2norm(y_11)

        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5, 
                    out_6, out_7, out_8, out_9, out_10, out_11), (
                    y_0, y_1, y_2, y_3, y_4, y_5,
                    y_6, y_7, y_8, y_9, y_10, y_11), maps, ids , avgs
        else:
            x = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11), 1)

            return  x, y

            
if __name__ == '__main__':
 net = embed_net(512, 319)
# 
 inx=torch.rand(2,3,288,144)
 flag = 0
 if flag:
     net.eval()
     out, y = net(inx, inx,1)
     print(len(y))
     for i in out:
         print(i.shape)
     for i in y:
         print(i.shape)
         
     out, y = net(inx, inx,2)
     print(len(y))
     for i in out:
         print(i.shape)
     for i in y:
         print(i.shape)
 else:         
     net.train()
     out, y, z, ids, avgs = net(inx, inx,0)
     print(len(y))
     for i in out:
         print(i.shape)
     for i in y:
         print(i.shape)
     print(z.shape)
     print(ids.shape)
     for i in avgs:
         print(i.shape)