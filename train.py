import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import numpy as np
import argparse
from tqdm import tqdm
# from ISDA_GAN import EstimatorCV, ISDALoss
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation,cutout'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
import os
from pytorch_model_summary import  summary
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tensorboardX import SummaryWriter

#torch.backends.cudnn.benchmark = True
class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        # batchsize = 4
        C = self.class_num
        # class = 2
        A = features.size(1)
        # A = 512
        # if labels ==1:
        #     one_hot = torch.ones_like(torch.tensor([4,1]))
        # else:
        #     one_hot = torch.zeros_like(torch.tensor([4,1]))
        # one_hot = torch.zeros_like(torch.tensor([4,1]))
        one_hot = torch.zeros(N, C).cuda()
        labels = labels.cuda()
        # (4,2)
        one_hot.scatter_(1,labels.view(-1,1),1)
        NxCxA_onehot = one_hot.view(N, C, 1).expand(N, C, A)
        # (4,2,512)
        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        # （4,1,512）--(4,2,512)

       #  one_hot = labels.cuda()
        # (4,2)
        # onehot.scatter_(1, labels.view(-1, 1), 1)


        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        # (10,640)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA
        #(10.640)
        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0),var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = one_hot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = one_hot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A) )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(torch.bmm((self.Ave - ave_CxA).view(C, A, 1),(self.Ave - ave_CxA).view(C, 1, A)))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += one_hot.sum(0)
        return self.CoVariance


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)


        self.class_num = class_num
    #    print(self.class_num)

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)
#         ratio = ratio.cuda()
        labels = labels.cuda()

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))

        CV_temp = cv_matrix[labels]
        # (128,640,640)
        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, fc, x, target_x, ratio):
        # x : Input value
        # target_x: label of x, 类别

        features = model(x)

        y = fc(features)

        self.estimator.update_CV(features.detach(), target_x)

        isda_aug_y = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

        loss = self.cross_entropy(isda_aug_y, target_x)

        return loss, y

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, fc, ratio, label, data_rf):
    """Train function of discriminator"""
    criteria = nn.CrossEntropyLoss()
    if data_rf == "real" :
        features, [rec_all, rec_small, rec_part], part, _ = net(data, data_rf="real")
        # print(data.size())
        # print(data_rf)
        # print(summary(net,[data, data_rf]))
        y = fc(features)
        feature_num = 512
        class_num = 2
        ratio = ratio
        label = label.cuda()
        estimator = EstimatorCV(feature_num, class_num)
        #EstimatorCV.update_CV(features=features.detach(), labels=label)
        Covariance = estimator.update_CV(features=features,labels=label)
        ISDALoss_1 = ISDALoss(feature_num,class_num)
        isda_aug_y = ISDALoss_1.isda_aug(fc, features, y, label, Covariance.detach(), ratio)
        loss_aug = criteria(isda_aug_y, label)
        # loss = loss_aug + F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() +\
        #     percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
        #     percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
        #     percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        loss = loss_aug +\
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        
       #  err = F.binary_cross_entropy(torch.ones_like([torch.Tensor(datas) for datas in data]),data)
       #  err.requires_grad_(True)
        loss.backward()
       # print("real success!")
        return loss.mean().item(), rec_all, rec_small, rec_part
    else:
        features, _ = net(data, data_rf="fake")
        y = fc(features)
        # EstimatorCV.update_CV(features.detach(), label)
        feature_num = 512
        class_num = 2
        label = label.cuda()
        estimator = EstimatorCV(feature_num, class_num)
        Covariance = estimator.update_CV(features=features, labels=label)
        # EstimatorCV.update_CV(features=features.detach(), labels=label)
        ISDALoss_2 = ISDALoss(feature_num, class_num)
        isda_aug_y = ISDALoss_2.isda_aug(fc, features, y, label, Covariance.detach(), ratio)
        loss_aug = criteria(isda_aug_y, label)
        loss = loss_aug
        loss.backward()
        # err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        #err = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred))
      #  err = F.binary_cross_entropy(torch.zeros_like(torch.Tensor(data)),data)
      #  err.requires_grad_(True)
        #err.backward()
        return loss_aug.mean().item()
        

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 1000
    saved_model_folder, saved_image_folder = get_dir(args)
    global class_num
    class_num = 2
    feature_num = 512
    fc = Full_layer(feature_num,class_num)
    isda_criterion = ISDALoss(feature_num, class_num).cuda()
    criteria = nn.CrossEntropyLoss()

    # ce_criterion = nn.CrossEntropyLoss.cuda()
    fc = nn.DataParallel(fc).cuda()

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

   # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   # print(device)

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    total_params_g = sum(p.numel() for p in netG.parameters())
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    total_params_d = sum(p.numel() for p in netD.parameters())
    netD.apply(weights_init)
    total_params = total_params_d + total_params_g
    # print(total_params)
    # print(total_params_d)
    # print(total_params_g)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    # noise_z = torch.FloatTensor(8, nz).normal_(0, 1)
    # noisetensor = torch.Tensor(8, nz)
    # noise_zin = torch.nn.init.uniform_(tensor=noisetensor, a=-1, b=1)
    # noise_zsig = torch.nn.init.constant_(tensor=noisetensor, val=0.2)
    # fixed_noise = torch.add(noise_zin, torch.mul(noise_z, noise_zsig)).to(device)
    # noise_z = torch.FloatTensor(8, nz).normal_(0, 1)
    # noisetensor = torch.Tensor(8, nz)
    # noise_zin = torch.nn.init.uniform_(tensor=noisetensor,a=-0.1, b=0.1)
    # noise_zsig = torch.nn.init.constant_(tensor=noisetensor, val=0.90)
    # noise_zuni = torch.ones(8, nz) - noise_zsig
    # fixed_noise = torch.add(torch.mul(noise_zuni, noise_zin), torch.mul(noise_z, noise_zsig)).to(device)
    # fixed_noise = torch.add(noise_zin, torch.mul(noise_z, noise_zsig)).to(device)
    #fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    if multi_gpu:
        netG = nn.DataParallel(netG.cuda())
        netD = nn.DataParallel(netD.cuda())

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.cuda(non_blocking=True)
        current_batch_size = real_image.size(0)
       # print(current_batch_size)
        #ratio = iteration / total_iterations
        ratio = 1.0
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        # noise_z = torch.FloatTensor(current_batch_size, nz).normal_(0, 1)
        # noisetensor = torch.Tensor(current_batch_size, nz)
        # noise_zin = torch.nn.init.uniform_(tensor=noisetensor, a=-1, b=1)
        # noise_zsig = torch.nn.init.constant_(tensor=noisetensor, val=0.2)
        # noise = torch.add(noise_zin, torch.mul(noise_z, noise_zsig)).to(device)
        # noise_z = torch.FloatTensor(current_batch_size, nz).normal_(0, 1)
        # noisetensor = torch.Tensor(current_batch_size, nz)
        # noise_zin = torch.nn.init.uniform_(tensor=noisetensor, a=-0.1, b=0.1)
        # noise_zsig = torch.nn.init.constant_(tensor=noisetensor, val=0.90)
        # noise_zuni = torch.ones(current_batch_size, nz) - noise_zsig
        # noise = torch.add(torch.mul(noise_zuni,noise_zin), torch.mul(noise_z, noise_zsig)).to(device)

       # noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        fake_images = netG(noise)
#       fake_images = fake_images.cuda(non_blocking=True)
       # print(summary(netG, noise))
        real_image = DiffAugment(real_image, policy=policy)
        # real_image = torch.FloatTensor(real_image.cpu().detach().numpy()).to(device)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]


        
        ## 2. train Discriminator
        netD.zero_grad()

        # def train_d(net, data, fc, ratio, label=1):
        lable_tenor = torch.randn([current_batch_size,])
        lable_tenor = lable_tenor.long()
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image,fc,ratio=ratio,label=torch.ones_like(lable_tenor),data_rf="real")

       # err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")

        train_d(netD, [fi.detach() for fi in fake_images], fc, ratio=ratio, label=torch.zeros_like(lable_tenor),data_rf="fake")
       # print(summary(netD, noise))
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()

        feature_g, _ = netD(fake_images, data_rf="fake")
        lable_tenor = torch.randn([current_batch_size, ])
        lable_tenor = lable_tenor.long()
        label = torch.zeros_like(torch.tensor(lable_tenor)).cuda()
        y = fc(feature_g)
        estimator = EstimatorCV(feature_num, class_num)
        Covariance = estimator.update_CV(features=feature_g, labels=label)
        ISDALoss_3 = ISDALoss(feature_num, class_num)
        isda_aug_y = ISDALoss_3.isda_aug(fc, feature_g, y, label, Covariance.detach(), ratio)
        # isda_aug_y = ISDALoss.isda_aug(fc, feature_g, y, label, CoVariance.detach(), ratio)
        loss_aug = - criteria(isda_aug_y, label)
        loss_aug.backward()


        #err_g = -F.binary_cross_entropy_with_logits(pred_g,  torch.zeros_like(pred_g))
        #sig_loss = 0.1 * torch.mean(np.square(noise_zsig - 1))
        #err_g = -pred_g.mean() + sig_loss

        # err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 10 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f    Total loss: %.5f"%(err_dr, -loss_aug.item(), err_dr-loss_aug.item()))

        if iteration % (save_interval) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (save_interval*10) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./datasets/100-shot-panda/img', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='Panda256', help='experiment name')
    parser.add_argument('--iter', type=int, default=100000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')


    args = parser.parse_args()
    print(args)

    train(args)