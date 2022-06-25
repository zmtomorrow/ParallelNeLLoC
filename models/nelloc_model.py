import torch.nn as nn
import torch
import torch.nn.functional as F

def discretized_mix_logistic_uniform(x, l, alpha=0.0001):
    xs=list(x.size())
    x=x.unsqueeze(2)
    mix_num = int(l.size(1)/10) 
    pi = torch.softmax(l[:, :mix_num,:,:],1).unsqueeze(1).repeat(1,3,1,1,1)
    l=l[:, mix_num:,:,:].view(xs[:2]+[-1]+xs[2:])
    means = l[:, :, :mix_num, :,:]
    inv_stdv = torch.exp(-torch.clamp(l[:, :, mix_num:2*mix_num,:, :], min=-7.))
    coeffs = torch.tanh(l[:, :, 2*mix_num:, : ,  : ])
    m2 = means[:,  1:2, :,:, :]+coeffs[:,  0:1, :,:, :]* x[:, 0:1, :,:, :]
    m3 = means[:,  2:3, :,:, :]+coeffs[:,  1:2, :,:, :] * x[:, 0:1,:,:, :]+coeffs[:,  2:3,:,:, :] * x[:,  1:2,:,:, :]
    means = torch.cat((means[:, 0:1,:, :, :],m2, m3), dim=1)
    centered_x = x - means
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))
    cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min = torch.sigmoid(inv_stdv * (centered_x - 1. / 255.))
    cdf_min=torch.where(x < -0.999, torch.tensor(0.0).to(x.device),cdf_min)
    log_probs =torch.log((1-alpha)*(pi*(cdf_plus-cdf_min)).sum(2)+alpha*(1/256))
    return -log_probs.sum([1,2,3]).mean()


class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = torch.zeros(1)
            self.mask[:,:,height//2+1:,:] = torch.zeros(1)
        else:
            self.mask[:,:,height//2,width//2+1:] = torch.zeros(1)
            self.mask[:,:,height//2+1:,:] = torch.zeros(1)


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedCNN, self).forward(x)
    

class LocalPixelCNN(nn.Module):
    def __init__(self, res_num=10, in_kernel = 7,  in_channels=3, channels=256, out_channels=256, device=None):
        super(LocalPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num
        

        self.in_cnn=MaskedCNN('A',in_channels,channels, in_kernel, 1, in_kernel//2, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
 
        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)

    
    def forward(self, x, train=True,rs=None):            
        x=self.in_cnn(x)
        if train==False:
            x=x[:,:,-1:,rs:rs+1]
        x=self.activation(x)

        for i in range(0, self.res_num):
            x_mid=self.resnet_cnn11[i](x)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn3[i](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn12[i](x_mid)
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.out_cnn1(x)
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x