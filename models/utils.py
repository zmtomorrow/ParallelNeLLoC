import torchvision
from torchvision import transforms
import torch
import numpy as np
from torch.utils import data

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def shear(x,offset=2):
    bs=x.size(0)
    D=x.size(2)
    L=D+(D-1)*offset
    sheared_img=torch.zeros(bs,3,D,L)
    for i in range(0,D):
        sheared_img[:,:,i:i+1,offset*i:offset*i+D]=x[:,:,i:i+1,:]
    return sheared_img

def shear_inv(sheared_x,offset=2):
    bs=sheared_x.size(0)
    D=sheared_x.size(2)
    o_x=torch.zeros(bs,3,D,D)
    for i in range(0,D):
        o_x[:,:,i:i+1,:]=sheared_x[:,:,i:i+1,offset*i:offset*i+D]
    return o_x

def shear_quantity(D,O):
    T=D+(D-1)*O
    t_vec=np.arange(0,T)
    up=(np.maximum(t_vec+1-D,np.zeros_like(t_vec))+O-1)//O
    down=(D-(np.maximum(T-t_vec-D,np.zeros_like(t_vec))+O-1)//O)
    bs=down-up
    return (D,O,T,up,down,bs)

def get_test_image(D,num=10,PATH = "./imgnet-small"):
    TRANSFORM_IMG = transforms.Compose([
        torchvision.transforms.Resize(D),
        transforms.CenterCrop(D),
        transforms.ToTensor(),
        ])
    test_data = torchvision.datasets.ImageFolder(root=PATH, transform=TRANSFORM_IMG)
    img_loader = torch.utils.data.DataLoader(test_data, batch_size=num,shuffle = False)
    for i in img_loader:
        img_batch=(i[0]*255).to(torch.int32)
        break
    return img_batch

def get_device(opt,gpu_index):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        opt["device"] = torch.device("cuda:"+str(gpu_index))
        opt["if_cuda"] = True
    else:
        opt["device"] = torch.device("cpu")
        opt["if_cuda"] = False
    return opt


def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())
        
    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug']==True:
            trans=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor()])
        else:
            trans=torchvision.transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader,test_data_loader,train_data_evaluation

