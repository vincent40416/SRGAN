import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import vgg16
import math
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
#   WGAN

import os
import pickle
import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
# HR_train_data_path = 'C:/Users/User/Desktop/GAN_data_set/imagenet64_train/Imagenet64_train'
HR_train_data_path = './SRGAN_training_data/*'
test_data_path = './SRGAN_test_data/*'

batch_size = 64
input_channels = 3
hr_height = 128
lr_height = 32
n_critic = 1
n_critic_D = 1
clip_value = 0.02
epochs = 100
num_epochs = 201
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class ImageDataset(Dataset):
    def __init__(self, imgs, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

        self.files = imgs

    def __getitem__(self, index):
        img =  Image.fromarray(self.files[index].astype('uint8'), 'RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)

class TestImageDataset(Dataset):
    def __init__(self, imgs, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

        self.files = imgs

    def __getitem__(self, index):
        img =  Image.fromarray(self.files[index].astype('uint8'), 'RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)
def load_databatch(data_folder, idx):
    data_file_HR = os.path.join(HR_train_data_path, 'train_data_batch_')
    d_HR = unpickle(data_file_HR + str(idx))
    x_HR = d_HR['data'][:4000]

    data_size = x_HR.shape[0]

    hr_height2 = hr_height * hr_height
    x_HR = np.dstack((x_HR[:, :hr_height2], x_HR[:, hr_height2:2*hr_height2], x_HR[:, 2*hr_height2:]))
    x_HR = x_HR.reshape((x_HR.shape[0], hr_height, hr_height, 3))

    lr_transforms = [
                    transforms.Resize((hr_height//4, hr_height//4), Image.BICUBIC),
                    transforms.ToTensor() ]

    hr_transforms = [
                    transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                    transforms.ToTensor() ]

    train_loader = torch.utils.data.DataLoader(ImageDataset(x_HR, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                        batch_size=batch_size, shuffle=True)
    return train_loader

def load_jpg(data_folder,batch_size = batch_size, shuffle=True):
#     print(data_folder)
    image_list = []
    print("start loading")
    for filename in glob.glob(data_folder): #assuming gif
        im = Image.open(filename)
        im = im.resize((hr_height, hr_height), Image.BICUBIC)
        im = np.array(im)
        if im.shape == (hr_height ,hr_height ,3):
            image_list.append(im)

    image = np.asarray(image_list)

    lr_transforms = [
                    transforms.Resize((lr_height, lr_height), Image.BICUBIC),
                    transforms.ToTensor()
                    ]

    hr_transforms = [
                    transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                    transforms.ToTensor()
                    ]

    train_loader = torch.utils.data.DataLoader(TestImageDataset(image, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                        batch_size=batch_size, shuffle=shuffle)

    return train_loader

dataloader = load_jpg(HR_train_data_path)
test_data_set = load_jpg(test_data_path,shuffle = False)
# print(dataloader)
print("end")

# Loss function
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
# def generator_loss(logits_fake):
#     size = logits_fake.shape[0]
#     true_labels = Variable(torch.ones(size, 1)).float().cuda()
#     loss = torch.mean(logits_fake)
#     return loss
# def adversarial_loss(logits_real, logits_fake): # 判别器的 loss
#     size = logits_real.shape[0]
#     true_labels = Variable(torch.ones(size, 1)).float().cuda()
#     false_labels = Variable(torch.zeros(size, 1)).float().cuda()
#     loss = torch.nn.L1Loss(logits_real, true_labels) + torch.nn.L1Loss(logits_fake, false_labels)
#     return loss

generator_loss = GeneratorLoss().cuda()
discriminator_loss = torch.nn.MSELoss()
content_loss = nn.MSELoss(size_average = True)



#Optimizer



def psnr_cal(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )

    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_lr = Tensor(batch_size, input_channels, lr_height, lr_height).cuda()
input_hr = Tensor(batch_size, input_channels, hr_height, hr_height).cuda()
valid = Variable(torch.ones(batch_size, 1)).float().cuda()
fake = Variable(torch.zeros(batch_size, 1)).float().cuda()

# def debug_memory():
#     import collections, gc, torch
#     tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
#                                   for o in gc.get_objects()
#                                   if torch.is_tensor(o))
#     for line in sorted(tensors.items()):
#         print('{}\t{}'.format(*line))
# debug_memory()

def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, generator_loss, sample_interval = 50, checkpoint_interval = 10):
    iter_count = 0
    current_batch = 0
    index = 0
    torch.cuda.empty_cache()
    for epoch in range(epochs, num_epochs):
        for i, img in enumerate(dataloader):
            if i == len(dataloader)-1:
                break;
            # model input
            imgs_lr = Variable(input_lr.copy_(img['lr'])).cuda()
            imgs_hr = Variable(input_hr.copy_(img['hr'])).cuda()
#             print(imgs_lr.size())

            fake_images = G_net(imgs_lr).detach()
            # Train Discriminator
#             if i %  n_critic_D == 0 :
            D_optimizer.zero_grad()

            #loss of real and fake image
            # Total loss

            # GAN

            # loss_D = discriminator_loss(D_net(imgs_hr), valid) + discriminator_loss(D_net(fake_images), fake)

            # WGAN
            loss_D = 1 - D_net(imgs_hr).mean() + D_net(fake_images).mean() # 判别器的 loss

            # RGAN
            # loss_D = -torch.mean(torch.log(torch.sigmoid(-Relative)))  # 判别器的 loss

            loss_D.backward()
            D_optimizer.step()



            # train generators
#             if i % n_critic == 0:
            G_optimizer.zero_grad()
            # Generate Hr from low resolution input
            # adversarial loss
            gen_img =  G_net(imgs_lr)
            # GAN
            # gen_error = generator_loss(D_net(fake_images), valid)
            # WGAN
#             gen_error = -torch.mean(D_net(gen_img))
#             content_error = content_loss(gen_img , imgs_hr)


            # VGG_feature_loss
#                 gen_features = feature_extractor(fake_images)
#                 real_features = Variable(feature_extractor(imgs_hr).data, requires_grad=False)
#                 VGG_loss = content_loss(real_features , gen_features)
    #             print(imgs_hr.size())


            #Total loss
            # GAN
            real_out = D_net(imgs_hr).mean()
            fake_out = D_net(fake_images).mean()
            loss_G = generator_loss(fake_out, gen_img, imgs_hr)

            # WGAN
            # loss_G = 1e-2*gen_error + content_error

            # RGAN
            # Relative = D_net(fake_images)-D_net(imgs_hr)

            loss_G.backward()
            G_optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                            (epoch, num_epochs, i, len(dataloader),
                                                            loss_D.item(), loss_G.item()))


#             print(fake_images.data.cpu().numpy())
            # Save image sample turn to next batch
            if i % sample_interval == 0 :
                index = index+1
                psnr_val = 0
                for i , test_img in enumerate(test_data_set):
                    if i >= 1:
                        break;
#                     print(test_img)
                    test_imgs_lr = Variable(input_lr.copy_(test_img['lr'])).detach()
                    test_imgs_hr = Variable(input_hr.copy_(test_img['hr'])).detach()
                    test_fake_images = G_net(test_imgs_lr)

                    psnr_val += psnr_cal(test_fake_images.data.cpu().numpy(),test_imgs_hr.data.cpu().numpy())
                    print("[PSNR: %f]"%psnr_val)
#                     save_image(test_imgs_lr,"C:/Users/User/Desktop/Deep_Learning/GAN_images/TESTIMAGE'{0}'.png".format(i+(epoch)*batch_size), normalize=True)
                save_image(torch.cat((test_fake_images.data, test_imgs_hr.data), -2),
                                "./SRGAN_save_image/'{0}'.png".format(i+(epoch)*len(dataloader)), normalize=True)

#             if i == (len(dataloader)-1) :
#                 break;

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_net.state_dict(), './SRGAN_save_model/generator_%d-RGAN.pth' % epoch)
            torch.save(D_net.state_dict(), './SRGAN_save_model/discriminator_%d-RGAN.pth' % epoch)

D = Discriminator().cuda()
G = Generator(4).cuda()
def get_optimizer_G(net):
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
    return optimizer
def get_optimizer_D(net):
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
    return optimizer
D_optim = get_optimizer_D(D)
G_optim = get_optimizer_G(G)

if epochs != 0:
    # Load pretrained models
    D.apply(weights_init_normal)
#    D.load_state_dict(torch.load('./SRGAN_save_model/discriminator_%d-RGAN.pth'% epochs))
    G.load_state_dict(torch.load('./SRGAN_save_model/generator_%d-RGAN.pth'% epochs))
else:
    # Initialize weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
train_a_gan(D, G, D_optim, G_optim, generator_loss)
