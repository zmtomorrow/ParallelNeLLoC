import torch.nn.functional as F
from coders.coder_utils import *


def p_ans_compression(model,img,time_index,h,w,rf,p_prec=16):
    c_list=[]
    p_list=[]
    p2d = (rf, rf, rf, 0)
    img = F.pad(img, p2d, "constant", 0)
    with torch.no_grad():
        for t,par_index_list in enumerate(time_index):
            patch_list=[]
            pixel_list=[]
            for i,j in par_index_list:
                patch_list.append(img[0,:,i:i+rf+1,j:j+rf+rf+1]/255.)
                pixel_list.append(img[0,:,i+rf,j+rf])
            
            bs=len(pixel_list)
            patches=torch.stack(patch_list)
            pixels=torch.stack(pixel_list).view(bs,3)
            
            model_outputs=model(rescaling(patches),False,rf)
            means,coeffs,log_scales, pi=compute_stats(model_outputs.view(bs,-1))

            for c in range(0,3):  
                if c==0:
                    mean=means[:,0:1,:]
                elif c==1:
                    c_0=rescaling(pixels[:,0:1]/255.).unsqueeze(-1)
                    mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_0
                else:
                    c_1=rescaling(pixels[:,1:2]/255.).unsqueeze(-1)
                    mean=means[:,2:3, :] + coeffs[:,1:2, :]* c_0 +coeffs[:,2:3, :] * c_1
                cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                c_list.extend(np.take_along_axis(cdf_min_table,pixels[:,c:c+1].numpy(),axis=-1).reshape(-1))
                p_list.extend(np.take_along_axis(probs_table,pixels[:,c:c+1].numpy(),axis=-1).reshape(-1)) 

    ans_stack=ANSStack(s_prec = 32,t_prec = 16, p_prec=p_prec)              
    for i in np.arange(len(c_list)-1,-1,-1):
        c_min,p=c_list[i],p_list[i]
        ans_stack.push(c_min,p)
    return ans_stack                



def p_ans_decompression(model,ans_stack,time_index,h,w,rf,p_prec=16):
    with torch.no_grad():
        decode_img=torch.zeros([1,3,h+2*rf,w+2*rf])
        for t,par_index_list in enumerate(time_index):
            patch_list=[]
            for i,j in par_index_list:
                patch_list.append(decode_img[0,:,i:i+rf+1,j:j+rf+rf+1]/255.)
            patches=torch.stack(patch_list)
            bs=len(patch_list)
            model_outputs=model(rescaling(patches),False,rf)
            means,coeffs,log_scales, pi=compute_stats(model_outputs.view(bs,-1))
            decoded_batch=torch.zeros([bs,3])
            for c in range(0,3): 
                if c==0:
                    mean=means[:,0:1, :]
                elif c==1:
                    c_0=rescaling(decoded_batch[:,0:1]/255.).unsqueeze(-1)
                    mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_0
                else:
                    c_1=rescaling(decoded_batch[:,1:2]/255.).unsqueeze(-1)
                    mean=means[:,2:3, :] + coeffs[:,1:2, :]* c_0 +coeffs[:,2:3, :] * c_1
                cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                for ind in range(0,bs):
                    s_bar = ans_stack.pop()
                    pt=np.searchsorted(cdf_min_table[ind], s_bar, side='right', sorter=None)-1
                    decoded_batch[ind,c]=int(pt)
                    cdf,p=int(cdf_min_table[ind][pt]),int(probs_table[ind][pt])
                    ans_stack.update(s_bar,cdf,p)
                    decode_img[0,c,par_index_list[ind][0]+rf,par_index_list[ind][1]+rf]=int(pt)

        return decode_img[:,:,rf:h+rf,rf:w+rf]
                            