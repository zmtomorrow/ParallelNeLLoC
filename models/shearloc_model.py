import torch.nn as nn
import torch



def discretized_mix_logistic_uniform(x, l,sheared_mask, alpha=0.0001):
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
    return -(log_probs*sheared_mask).sum([1,2,3]).mean()





class LocalPixelCNN(nn.Module):
    def __init__(self, res_num=10, kernel_size = [2,1],  in_channels=3, channels=256, out_channels=256):
        super(LocalPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.res_num=res_num

        self.in_cnn=nn.Conv2d(in_channels,channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([nn.Conv2d(channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([nn.Conv2d(channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([nn.Conv2d(channels,channels, 1, 1, 0) for i in range(0,res_num)])
 
        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)

    def forward(self,x,train=True,up=None,down=None):
        x=self.in_cnn(x)
        if train==False:
            x=x[:,:,:,-1:]
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

    # def forward(self, x):
    #     x=self.in_cnn(x)
    #     x=self.activation(x)

    #     for i in range(0, self.res_num):
    #         x_mid=self.resnet_cnn11[i](x)
    #         x_mid=self.activation(x_mid)
    #         x_mid=self.resnet_cnn3[i](x_mid)
    #         x_mid=self.activation(x_mid)
    #         x_mid=self.resnet_cnn12[i](x_mid)
    #         x_mid=self.activation(x_mid)
    #         x=x+x_mid
    #     x=self.out_cnn1(x)
    #     x=self.activation(x)
    #     x=self.out_cnn2(x)
    #     return x


