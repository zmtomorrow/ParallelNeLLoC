import torch
import numpy as np


rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5




def discretized_mix_logistic_cdftable(means, log_scales,pi, alpha=0.0001):
    bs=means.size(0)
    nr_mix=pi.size(-1)
    pi=pi.unsqueeze(1)
    x=rescaling(torch.arange(0,256)/255.).view(1,256,1).repeat(bs,1,nr_mix)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))
    cdf_min = torch.sigmoid(inv_stdv * (centered_x - 1. / 255.))
    mix_cdf_plus=(pi*cdf_plus).sum(-1)
    mix_cdf_min=(pi*cdf_min).sum(-1)
    return mix_cdf_plus,mix_cdf_min

    # cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    # cdf_min=torch.where(x <- 0.999, torch.tensor(0.0).to(x.device),cdf_min)

    # uniform_cdf_min = ((x+1.)/2*255)/256.
    # uniform_cdf_plus = ((x+1.)/2*255+1)/256.
    

    # mix_cdf_plus=((1-alpha)*pi*cdf_plus+(alpha/10)*uniform_cdf_plus).sum(-1)
    # mix_cdf_min=((1-alpha)*pi*cdf_min+(alpha/10)*uniform_cdf_min).sum(-1)
    # return mix_cdf_plus,mix_cdf_min


def compute_stats(l):
    bs=l.size(0)
    nr_mix=int(l.size(1)/10)
    pi=torch.softmax(l[:,:nr_mix],-1)
    l=l[:,nr_mix:].view(bs,3,-1)
    means=l[:,:,:nr_mix]
    log_scales = torch.clamp(l[:,:,nr_mix:2 * nr_mix], min=-7.)
    coeffs = torch.tanh(l[:,:,2 * nr_mix:3 * nr_mix])
    return means,coeffs,log_scales, pi 

def get_mean_c1(means,mean_linear,x):
    return means+x.unsqueeze(-1)*mean_linear

def get_mean_c2(means,mean_linear,x):
    print(means.size())
    return means+torch.bmm(x.view(-1,1,2),mean_linear.view(-1,2,10)).view(-1,1,10)


def cdf_table_processing(cdf_plus,cdf_min,p_prec):
    p_total=np.asarray((1 << p_prec),dtype='uint32')
    bs=cdf_plus.size(0)
    cdf_min=np.rint(cdf_min.numpy()*  p_total).astype('uint32')
    cdf_plus=np.rint(cdf_plus.numpy()* p_total).astype('uint32')
    probs=cdf_plus-cdf_min
    probs[probs==0]=1
    argmax_index=np.argmax(probs,axis=1).reshape(-1,1)
    diff=p_total-np.sum(probs,-1,keepdims=True)
    value=diff+np.take_along_axis(probs, argmax_index.reshape(-1,1), axis=-1)
    np.put_along_axis(probs, argmax_index,value , axis=-1)
    return np.concatenate((np.zeros((bs,1),dtype='uint32'),np.cumsum(probs[:,:-1],axis=-1,dtype='uint32')),1),probs

def ians_get_length(s,t_stack):
    return  len(t_stack)*len(bin(t_stack[0]))+sum(len(bin(i)) for i in s)

class ANSStack(object):
    def __init__(self, s_prec , t_prec, p_prec):
        self.s_prec=s_prec
        self.t_prec=t_prec
        self.p_prec=p_prec
        self.t_mask = (1 << t_prec) - 1
        self.s_min=1 << s_prec - t_prec
        self.s_max=1 << s_prec
        self.s, self.t_stack= self.s_min, [] 

    def push(self,c_min,p):
        while self.s >= p << (self.s_prec - self.p_prec):
            self.t_stack.append(self.s & self.t_mask )
            self.s=self.s>> self.t_prec
        self.s = (self.s//p << self.p_prec) + self.s%p + c_min
        assert self.s_min <= self.s < self.s_max

    def pop(self):
        return self.s & ((1 << self.p_prec) - 1)

    def update(self,s_bar,c_min,p):
        self.s = p * (self.s >> self.p_prec) + s_bar - c_min
        while self.s < self.s_min:
            t_top=self.t_stack.pop()
            self.s = (self.s << self.t_prec) + t_top
        assert self.s_min <= self.s < self.s_max
        
    def get_length(self):
        return len(self.t_stack)*self.t_prec+len(bin(self.s))