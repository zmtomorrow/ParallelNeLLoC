import torch
from model import *
from decimal import *

tensor2decimal= lambda x : Decimal(str(x.cpu().item()))


def bin_2_float(binary):
    prob = Decimal('0.0')
    cur_prob=Decimal('0.5')
         
    for i in binary:
        prob=prob+cur_prob* int(i)
        cur_prob*=Decimal('0.5')
    return prob


def range_2_bin(low, high):
    code = []
    prob = Decimal('0.0')
    cur_prob=Decimal('0.5')
    
    while(prob < low):
        acc_prob=prob+cur_prob
        if acc_prob > high:
            code.append(0)
        else:
            code.append(1)
            prob = acc_prob
        cur_prob*=Decimal('0.5')
    return code



def logistic_autoreg_cdftable(means,log_scales,alpha=0.0005):
    x_t=torch.arange(0,256)
    centered_x =  x_t/255.- means   
    inv_stdv = torch.exp(-torch.clamp(log_scales,min=-7.))
    cdf_plus = (1-alpha)*torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))+alpha*(x_t+1)/256. 
    cdf_min = (1-alpha)*torch.sigmoid(inv_stdv * centered_x)+alpha*x_t/256.    
    return cdf_plus,cdf_min


calculate_means= lambda means, mean_linear,x: means+x@torch.tanh(mean_linear)

criterion_list = [
    lambda x, p,m,n: logistic_autoreg_cdftable(p[0,0,m,n],p[0,3,m,n]), \
    lambda x, p,m,n: logistic_autoreg_cdftable(*calculate_means(p[0,1:2,m,n],p[0,6:7,m,n],x[0,0:1,m,n]),p[0,4,m,n]),\
    lambda x, p,m,n: logistic_autoreg_cdftable(*calculate_means(p[0,2:3,m,n],p[0,7:9,m,n],x[0,0:2,m,n]),p[0,5,m,n])]


def ac_decompression_parallel(model,code,h,w,time_index,k=5):
    with torch.no_grad():
        model.eval()
        device=next(model.parameters()).device
        prob = bin_2_float(code)
        low = Decimal(0.0)
        high = Decimal(1.0)
        _range = Decimal(1.0)
        padding=int(k/2)
        mid=padding
        decode_img=torch.zeros([1,3,h+padding*2,w+padding*2])
        for par_index_list in time_index:
            patch_list=[]
            for i,j in par_index_list:
                patch_list.append(decode_img[0,:,i:i+mid+1,j:j+k]/255)
            patches=torch.stack(patch_list)

            model_outputs=model(patches.to(device))
            for patch_index,(i,j) in enumerate(par_index_list):
                patch=patches[patch_index:patch_index+1]
                model_output=model_outputs[patch_index:patch_index+1]

                for c in range(0,3):
                    cdf_plus_table,cdf_min_table= criterion_list[c](patch.to(device), model_output,mid,mid)
                    s=128
                    bl=0
                    br=256
                    for _ in range(0,9):
                        if  tensor2decimal(cdf_min_table[s])>prob:
                            br=s
                            s=int((s+bl)/2)
                        elif tensor2decimal(cdf_plus_table[s])<prob:
                            bl=s
                            s=int((s+br)/2)
                        else:
                            decode_img[0,c,i+2,j+2]=s
                            low=tensor2decimal(cdf_min_table[s])
                            high=tensor2decimal(cdf_plus_table[s])
                            _range=high-low
                            prob=(prob-low)/_range 
                            patch[0,c,mid,mid]=s/255. 
                            break
    return  decode_img[0,0:3,padding:h+padding,padding:w+padding]
    


def ac_compression_parallel(model,img,time_index,k=5):
    with torch.no_grad():
        padding=int(k/2)
        p2d = (padding, padding, padding, padding)
        img = F.pad(img, p2d, "constant", 0)     
        model.eval()
        device=next(model.parameters()).device
        mid=padding
        size=img.size()
        old_low  = Decimal('0.0')
        _range   = Decimal('1.0')
        for par_index_list in time_index:
            patch_list=[]
            for i,j in par_index_list:
                patch_list.append(img[0,:,i:i+mid+1,j:j+k]/255)
            patches=torch.stack(patch_list)
            model_outputs=model(patches.to(device))
            for patch_index in range(0,len(par_index_list)):
                patch=patches[patch_index:patch_index+1]
                model_output=model_outputs[patch_index:patch_index+1]
                for c in range(0,size[1]):
                    cdf_plus_table,cdf_min_table= criterion_list[c](patch.to(device), model_output,mid,mid) 
                    low  = old_low + _range * tensor2decimal(cdf_min_table[int(patch[0,c,mid,mid]*255)])
                    high = old_low + _range * tensor2decimal(cdf_plus_table[int(patch[0,c,mid,mid]*255)])
                    _range = high - low
                    old_low  = low
    print(low,high)
    code=range_2_bin(low,high)
    return code



def ac_decompression(model,code,h,w,k=5):
    with torch.no_grad():
        model.eval()
        device=next(model.parameters()).device
        prob = bin_2_float(code)
        low = Decimal(0.0)
        high = Decimal(1.0)
        _range = Decimal(1.0)
        rf=int(k/2)
        decode_img=torch.zeros([1,3,h,w])
        for i in range(0,h):
            for j in range(0,w):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                patch=decode_img[:,:,up:down,left:right]/255.
                m,n=min(rf,i),min(rf,j)
                model_output=model(patch)
                for c in range(0,3):
                    cdf_plus_table,cdf_min_table= criterion_list[c](patch.to(device), model_output,m,n)
                    s=128
                    bl=0
                    br=256
                    for _ in range(0,9):
                        if  tensor2decimal(cdf_min_table[s])>prob:
                            br=s
                            s=int((s+bl)/2)
                        elif tensor2decimal(cdf_plus_table[s])<prob:
                            bl=s
                            s=int((s+br)/2)
                        else:
                            decode_img[0,c,i,j]=s
                            low=tensor2decimal(cdf_min_table[s])
                            high=tensor2decimal(cdf_plus_table[s])
                            _range=high-low
                            prob=(prob-low)/_range 
                            patch[0,c,m,n]=s/255. 
                            break
        return  decode_img[0]



def ac_compression(model,img,k=5):
    with torch.no_grad():
        model.eval()
        device=next(model.parameters()).device
        rf=int(k/2)
        size=img.size()
        old_low  = Decimal('0.0')
        _range   = Decimal('1.0')
        for i in range(0,size[2]):
            for j in range(0,size[3]):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                m,n=min(rf,i),min(rf,j)
                patch=img[:,:,up:down,left:right]/255
                model_output=model(patch.to(device))
                for c in range(0,size[1]):
                    cdf_plus_table,cdf_min_table= criterion_list[c](patch.to(device), model_output,m,n) 
                    low  = old_low + _range * tensor2decimal(cdf_min_table[int(patch[0,c,m,n]*255)])
                    high = old_low + _range * tensor2decimal(cdf_plus_table[int(patch[0,c,m,n]*255)])
                    _range = high - low
                    old_low  = low
    code=range_2_bin(low,high)
    return code