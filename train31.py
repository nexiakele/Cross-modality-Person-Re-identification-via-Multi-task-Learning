from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData3, RegDBData2, TestData
from skimage import io
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model31 import embed_net
from utils import *
from loss.triplet import Cross_modal_ContrastiveLoss6
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, 
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, 
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only') 
parser.add_argument('--model_path', default='save_model/', type=str, 
                    help='model save path')
parser.add_argument('--save_epoch', default=1, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=6144, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(0)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data/RGBT-reid/SYSU_MM01/'
#    data_path ='/media/hpc/data/work/dataset/RGBT_Persion_ReID/SYSU_MM01/'

#    data_path ='/home/ly/RGT-id/SYSU-MM01/'

    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2] # thermal to visible
elif dataset =='regdb':
    data_path = '/data/RGBT-reid/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1] # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
suffix = 'model31_3'
if dataset =='regdb':
    suffix = suffix + '_regdb_trial_{}'.format(args.trial)

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
feature_dim = args.low_dim

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset =='sysu':
    # training set
    trainset = SYSUData3(data_path,  transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)
      
elif dataset =='regdb':
    # training set
    trainset = RegDBData2(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')

gallset  = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    
# testing data loader
gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
   
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')   
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))


print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch)
net.to(device)
cudnn.benchmark = True

if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

if args.method =='id':
    criterion = nn.CrossEntropyLoss().cuda()
    ##features_dim, num_class=10, margin=0.5, lamda=1., scale=1.0, batch_size=64
    triloss = Cross_modal_ContrastiveLoss6().cuda()
    seg_loss = nn.BCELoss().cuda()
    avg_loss = nn.MSELoss().cuda()
    ignored_params = list(map(id, net.shared.parameters())) \
                 + list(map(id, net.feature1.parameters())) \
                 + list(map(id, net.feature2.parameters())) \
                 + list(map(id, net.feature3.parameters())) \
                 + list(map(id, net.feature4.parameters())) \
                 + list(map(id, net.feature5.parameters())) \
                 + list(map(id, net.feature6.parameters())) \
                 + list(map(id, net.feature7.parameters())) \
                 + list(map(id, net.feature8.parameters())) \
                 + list(map(id, net.feature9.parameters())) \
                 + list(map(id, net.feature10.parameters())) \
                 + list(map(id, net.feature11.parameters())) \
                 + list(map(id, net.feature12.parameters())) \
                 + list(map(id, net.classifier1.parameters())) \
                 + list(map(id, net.classifier2.parameters())) \
                 + list(map(id, net.classifier3.parameters()))\
                 + list(map(id, net.classifier4.parameters()))\
                 + list(map(id, net.classifier5.parameters()))\
                 + list(map(id, net.classifier6.parameters()))  \
                 + list(map(id, net.classifier7.parameters())) \
                 + list(map(id, net.classifier8.parameters())) \
                 + list(map(id, net.classifier9.parameters()))\
                 + list(map(id, net.classifier10.parameters()))\
                 + list(map(id, net.classifier11.parameters()))\
                 + list(map(id, net.classifier12.parameters())) 
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
if args.optim == 'sgd':
    optimizer = optim.SGD([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': net.shared.parameters(), 'lr': args.lr},
         {'params': net.feature1.parameters(), 'lr': args.lr},
         {'params': net.feature2.parameters(), 'lr': args.lr},
         {'params': net.feature3.parameters(), 'lr': args.lr},
         {'params': net.feature4.parameters(), 'lr': args.lr},
         {'params': net.feature5.parameters(), 'lr': args.lr},
         {'params': net.feature6.parameters(), 'lr': args.lr},
         {'params': net.feature7.parameters(), 'lr': args.lr},
         {'params': net.feature8.parameters(), 'lr': args.lr},
         {'params': net.feature9.parameters(), 'lr': args.lr},
         {'params': net.feature10.parameters(), 'lr': args.lr},
         {'params': net.feature11.parameters(), 'lr': args.lr},
         {'params': net.feature12.parameters(), 'lr': args.lr},
         {'params': net.classifier1.parameters(), 'lr': args.lr},
         {'params': net.classifier2.parameters(), 'lr': args.lr},
         {'params': net.classifier3.parameters(), 'lr': args.lr},
         {'params': net.classifier4.parameters(), 'lr': args.lr},
         {'params': net.classifier5.parameters(), 'lr': args.lr},
         {'params': net.classifier6.parameters(), 'lr': args.lr},
         {'params': net.classifier7.parameters(), 'lr': args.lr},
         {'params': net.classifier8.parameters(), 'lr': args.lr},
         {'params': net.classifier9.parameters(), 'lr': args.lr},
         {'params': net.classifier10.parameters(), 'lr': args.lr},
         {'params': net.classifier11.parameters(), 'lr': args.lr},
         {'params': net.classifier12.parameters(), 'lr': args.lr},
         ],
         weight_decay=5e-4, momentum=0.9, nesterov=True)

elif args.optim == 'adam':
    optimizer = optim.Adam([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': net.feature.parameters(), 'lr': args.lr},
         {'params': net.classifier.parameters(), 'lr': args.lr}],weight_decay=5e-4)
         
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5:
        lr = args.lr
    elif epoch < 10:
        lr = args.lr * 0.75
    elif epoch < 15:
        lr = args.lr * 0.5
    elif epoch < 20:
        lr = args.lr * 0.25
    elif epoch < 25:
        lr = args.lr * 0.125
    elif epoch < 30:
        lr = args.lr * 0.1  
      
        
    print(len(optimizer.param_groups))
    optimizer.param_groups[0]['lr'] = 0.1*lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr
    optimizer.param_groups[5]['lr'] = lr
    optimizer.param_groups[6]['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr
    optimizer.param_groups[8]['lr'] = lr
    optimizer.param_groups[9]['lr'] = lr
    optimizer.param_groups[10]['lr'] = lr
    optimizer.param_groups[11]['lr'] = lr
    optimizer.param_groups[12]['lr'] = lr
    optimizer.param_groups[13]['lr'] = lr
    optimizer.param_groups[14]['lr'] = lr
    optimizer.param_groups[15]['lr'] = lr
    optimizer.param_groups[16]['lr'] = lr
    optimizer.param_groups[17]['lr'] = lr
    optimizer.param_groups[18]['lr'] = lr
    optimizer.param_groups[19]['lr'] = lr
    optimizer.param_groups[20]['lr'] = lr
    optimizer.param_groups[21]['lr'] = lr
    optimizer.param_groups[22]['lr'] = lr
    optimizer.param_groups[23]['lr'] = lr
    optimizer.param_groups[24]['lr'] = lr
    optimizer.param_groups[25]['lr'] = lr
    return lr
def save_rgb(img_tensor, dir_path,name):
    p=lambda out,i,name:io.imsave(name,out[i].transpose((1, 2, 0)))
    out = img_tensor.detach().numpy()
    l   = len(name)
    for i in range(l):
          p(out, i,dir_path+name[i]+'.jpg')
def save_d(img_tensor, dir_path,name):
    p=lambda out,i,j,name:io.imsave(name,out[i][j])
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    
    l   = len(name)
    for i in range(l):
          p(out, i,0,dir_path+name[i]+'.png')    
          
def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, (input1, input2, label1, label2, mask1, mask2) in enumerate(trainloader):
#        save_rgb(input1, './vis_mask_RegDB/', [str(batch_idx)+ str(i.item()) +'_rgb' for i in label1])
#        save_rgb(input2, './vis_mask_RegDB/', [str(batch_idx)+str(i.item()) +'_ir' for i in label1])
#        save_d(mask1, './vis_mask_RegDB/', [str(batch_idx)+str(i.item()) +'_RGB_mask' for i in label1])
#        save_d(mask2, './vis_mask_RegDB/', [str(batch_idx)+str(i.item()) +'_ir_mask' for i in label1])
        
        
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        labels = torch.cat((label1,label2),0).long()
        labels = Variable(labels.cuda()).long()
        labels1= Variable(label1.cuda()).long()
        

        data_time.update(time.time() - end)
        masks = torch.cat((mask1, mask2),0).cuda()
        
        
        

        outputs, feat, pmask, ids, avgs = net(input1, input2)
        loss_back1, loss_back2 = [], []
        loss_cuda = []
        if args.method =='id':
            loss1=0.0
            for i, out in enumerate(outputs):
                lo=criterion(out, labels) 
                loss_back2.append(lo)
                loss1+=lo
            loss2 = 0    
            for i, f in enumerate(feat):
                b = f.size(0)
                rfeat = f[0:b//2,:]
                tfeat = f[b//2:,:]
                lo, prec = triloss(rfeat, tfeat, labels1)
                lo =  lo  * 0.2
                loss2 += lo
                loss_back1.append(lo)
                loss_cuda.append(torch.tensor(1.0).cuda())
             
            ###########################################################################    
            b, c, h, w = masks.shape
            pmask = torch.nn.functional.interpolate(pmask, (h,w))
            loss3 = seg_loss(pmask, masks) * 5
            ###########################################################################
            loss4 =   criterion(ids, labels)  
            
            loss5 = 0 
            loss_back5 = []
            for i, f in enumerate(avgs):
                b = f.size(0)
                rfeat = f[0:b//2,:]
                tfeat = f[b//2:,:]
                lo  = avg_loss(rfeat, tfeat) 
                lo =  lo  * 5
                loss5 += lo
                loss_back5.append(lo)
                loss_cuda.append(torch.tensor(1.0).cuda())
             ###########################################################################
            _, predicted = outputs[0].max(1)
            correct += predicted.eq(labels).sum().item()
#            print(loss.item(), loss2.item(), loss3.item())
            loss=loss1+loss2+loss3+loss4+loss5
            
        ###########################################################################    
        optimizer.zero_grad()  
#        loss.backward()
        loss_back = []
        for (i,j) in zip(loss_back1, loss_back2):
            loss_back.append((i+j).cuda())
        ###########################################################################
        loss_back.append(loss3)
        loss_cuda.append(torch.tensor(1.0).cuda())  
        ###########################################################################
        loss_back.append(loss4)
        loss_cuda.append(torch.tensor(1.0).cuda())  
        loss_back = loss_back + loss_back5
        ###########################################################################
#        print(len(loss_back), len(loss_cuda))
        torch.autograd.backward(loss_back, loss_cuda)
        optimizer.step()
        train_loss.update(loss.item(), 2*input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{} '
                  'Accu: {:.2f}' .format(
                  epoch, batch_idx, len(trainloader),current_lr, 
                  100.*correct/total, batch_time=batch_time, 
                  data_time=data_time), 
                  'L1: {:.4f}, L2: {:.4f}, L3: {:.4f}, L4: {:.4f}, L5: {:.4f}'.format(loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))

def test(epoch):   
    # switch to evaluation mode
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, args.low_dim))
    gall_feat2 = np.zeros((ngall, args.low_dim//2))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            gall_feat2[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))   

    # switch to evaluation mode
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, args.low_dim))
    query_feat2 = np.zeros((nquery, args.low_dim//2))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            query_feat2[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    
    start = time.time()
    # compute the similarity
    distmat  = np.matmul(query_feat, np.transpose(gall_feat))
    distmat2  = np.matmul(query_feat2, np.transpose(gall_feat2))
    # evaluation
    if dataset =='regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
        cmc2, mAP2 = eval_regdb(-distmat2, query_label, gall_label)
    elif dataset =='sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time()-start))
    return cmc, mAP, cmc2, mAP2
    
# training
print('==> Start Training...')    
for epoch in range(start_epoch, 29):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler3(trainset.train_color_label, \
        trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size, 4)
    trainset.cIndex = sampler.index1 # color index
    trainset.tIndex = sampler.index2 # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,\
        sampler = sampler, num_workers=args.workers, drop_last =True )

    if epoch >= 1:
        print ('Test Epoch: {}'.format(epoch))
        print ('Test Epoch: {}'.format(epoch),file=test_log_file)
        # testing
        cmc, mAP,cmc2, mAP2  = test(epoch)

        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP), file = test_log_file)
        test_log_file.flush()
        
        print('Pool FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc2[0], cmc2[4], cmc2[9], mAP2))
        print('Pool FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc2[0], cmc2[4], cmc2[9], mAP), file = test_log_file)
        test_log_file.flush()
        
        # save model
        if cmc[0] > best_acc and epoch >0: # not the real best for sysu-mm01 
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
    
    train(epoch)

    if epoch == 30:
        epoch = epoch + 1
        print ('Test Epoch: {}'.format(epoch))
        print ('Test Epoch: {}'.format(epoch),file=test_log_file)
        # testing
        cmc, mAP,cmc2, mAP2  = test(epoch)

        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP), file = test_log_file)
        test_log_file.flush()
        
        print('Pool FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc2[0], cmc2[4], cmc2[9], mAP2))
        print('Pool FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc2[0], cmc2[4], cmc2[9], mAP), file = test_log_file)
        test_log_file.flush()
        
        # save model
        if cmc[0] > best_acc : # not the real best for sysu-mm01 
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
            
    # save model every 20 epochs    
    if epoch > 5 and epoch%args.save_epoch ==0:
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))