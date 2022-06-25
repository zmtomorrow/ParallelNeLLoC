import torch.nn.functional as F
from coders.coder_utils import *


def ans_compression(model,img,h,w,rf,p_prec=16):
    c_list=[]
    p_list=[]
    p2d = (rf, rf, rf, 0)
    img = F.pad(img, p2d, "constant", 0)
    with torch.no_grad():
        for i in range(0,h):
            for j in range(0,w):
                patch=img[:,:,i:i+rf+1,j:j+rf+rf+1]/255.
                model_output=model(rescaling(patch),False,rf)
                means,coeffs,log_scales, pi=compute_stats(model_output.view(1,-1))
                for c in range(0,3):  
                    if c==0:
                        mean=means[:,0:1,:]
                    elif c==1:
                        c_0=rescaling(int(img[0,0,i+rf,j+rf])/255.)
                        mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_0
                    else:
                        c_1=rescaling(int(img[0,1,i+rf,j+rf])/255.)
                        mean=means[:,2:3, :] + coeffs[:,1:2, :]* c_0 +coeffs[:,2:3, :] * c_1
                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                    pixel_value=int(patch[0,c,rf,rf]*255)
                    c_list.append(int(cdf_min_table[0][pixel_value]))
                    p_list.append(int(probs_table[0][pixel_value]))
    ans_stack=ANSStack(s_prec = 32,t_prec = 16, p_prec=p_prec)              
    for i in np.arange(len(c_list)-1,-1,-1):
        c_min,p=c_list[i],p_list[i]
        ans_stack.push(c_min,p)
    return ans_stack    
    

def ans_decompression(model,ans_stack,h,w,rf,p_prec=16):
    with torch.no_grad():
        decode_img=torch.zeros([1,3,h+2*rf,w+2*rf])
        for i in range(0,h):
            for j in range(0,w):
                patch=decode_img[:,:,i:i+rf+1,j:j+rf+rf+1]/255.
                model_output=model(rescaling(patch),False,rf)
                means,coeffs,log_scales, pi=compute_stats(model_output.view(1,-1))
                c_vector=[0,0,0]
                for c in range(0,3):
                    if c==0:
                        mean=means[:,0:1, :]
                    elif c==1:
                        mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_vector[0]
                    else:
                        mean=means[:, 2:3, :] + coeffs[:, 1:2, :]* c_vector[0] +coeffs[:, 2:3, :] *  c_vector[1]
                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                    s_bar = ans_stack.pop()
                    pt=np.searchsorted(cdf_min_table[0], s_bar, side='right', sorter=None)-1
                    decode_img[0,c,i+rf,j+rf]=pt
                    c_vector[c]=torch.tensor(rescaling(pt/255.))
                    c,p=int(cdf_min_table[0][pt]),int(probs_table[0][pt])
                    ans_stack.update(s_bar,c,p)
        return decode_img[0,:,rf:h+rf,rf:w+rf]