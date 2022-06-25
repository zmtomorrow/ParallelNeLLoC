import torch
from torch import optim
from nelloc_model import *
import numpy as np
from torch.optim import lr_scheduler
from utils import *

def train_model(opt):
    name='nelloc_res'+str(opt['res_num'])+'_mixnum'+str(opt['mix_num'])+'_rf'+str(opt['rf'])

    train_data_loader,test_data_loader,_=LoadData(opt)
    net = LocalPixelCNN(res_num=opt['res_num'], in_kernel = opt['rf']*2+1,  out_channels=opt['mix_num']*10).to(opt['device'])
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99995)
    criterion  = lambda real, fake : discretized_mix_logistic_uniform(real, fake, alpha=0.0001)
    
    test_list=[]
    for e in range(1,opt['epochs']+1):
        print('epoch',e)
        net.train()
        for images, _ in train_data_loader:
            images = rescaling(images).to(opt['device'])
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(images, output)
            loss.backward()
            optimizer.step()    
        scheduler.step()
        
        
        with torch.no_grad():
            net.eval()
            bpd_cifar_sum=0.
            for i, (images, labels) in enumerate(test_data_loader):
                images = rescaling(images).to(opt['device'])
                output = net(images)
                loss = criterion(images, output).item()
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
    opt['rf']=3
    train_model(opt)

