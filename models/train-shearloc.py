import torch
from torch import optim
from shearloc_model import *
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler
from utils import *


def train_model(opt):
    k_h=opt['k_h']
    k_w=opt['k_w']
    offset=opt['offset']
    name='shearloc_res'+str(opt['res_num'])+'_mixnum'+str(opt['mix_num'])+'_kh'+str(opt['k_h'])+'_kw'+str(opt['k_w'])+'_o'+str(opt['offset'])

    train_data_loader,test_data_loader,_=LoadData(opt)

    net = LocalPixelCNN( res_num=opt['res_num'], kernel_size = [k_h,k_w], out_channels=opt['mix_num']*10).to(opt['device'])

    mask=torch.ones(opt['batch_size'],3,32,32)
    sheared_mask=shear(mask,offset).to(opt['device'])

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99995)
    criterion  = lambda real, fake : discretized_mix_logistic_uniform(real, fake, sheared_mask, alpha=0.0001)

    p2d=[k_w,0,k_h-1,0]

    test_list=[]
    for e in range(1,opt['epochs']+1):
        net.train()
        for images, _ in train_data_loader:
            images = rescaling(images)
            sheared_images=shear(images,offset)
            images_padded = F.pad(sheared_images, p2d, "constant", 0.)
            optimizer.zero_grad()
            output = net(images_padded.to(opt['device']))[:,:,:,:-1]
            loss = criterion(sheared_images.to(opt['device']), output)
            loss.backward()
            optimizer.step()
        scheduler.step()

        
        with torch.no_grad():
            net.eval()
            bpd_cifar_sum=0.
            for i, (images, _) in enumerate(test_data_loader):
                images = rescaling(images)
                sheared_images=shear(images,offset)
                images_padded = F.pad(sheared_images, p2d, "constant", 0.)
                output = net(images_padded.to(opt['device']))[:,:,:,:-1]
                loss = criterion(sheared_images.to(opt['device']), output).item()
                bpd_cifar_sum+=loss/(np.log(2.)*(32*32*3))
            bpd_cifar=bpd_cifar_sum/len(test_data_loader)
            print('epoch',e,bpd_cifar)
            test_list.append(bpd_cifar)

        np.save(opt['result_path']+name,test_list)
        torch.save(net.state_dict(),opt['save_path']+name+'.pth')




if __name__ == "__main__":
    opt = {}
    opt=get_device(opt,gpu_index=str(0))
    opt['data_set']='CIFAR'
    opt['dataset_path']='../../data/cifar10'
    opt['save_path']='../save/'
    opt['result_path']='../results/'
    opt['data_aug']=True

    opt['epochs'] = 200
    opt['batch_size'] = 100
    opt['test_batch_size']=100
    opt['seed']=0


    opt['res_num']=3
    opt['mix_num']=5
    opt['k_h']=3
    opt['k_w']=5
    opt['offset']=2
    train_model(opt)
