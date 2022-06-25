import torch.nn.functional as F
from coders.coder_utils import *
from models.utils import *

def ans_compression(model,img,Q,K,p_prec=16):
    model.eval()        
    c_list=[]
    p_list=[]
    D,O,T,up_batch,down_batch,bs_batch=Q
    sheared_o_img=shear(img,O).to(torch.int32)
    kh,kw=K
    p2d=[kw,0,kh-1,0]
    padded_img = F.pad(shear(torch.zeros([1,3,D,D]),O), p2d, "constant", 0)
    with torch.no_grad():
        for t in range(0,T):
            up=up_batch[t]
            down=down_batch[t]
            bs=bs_batch[t]
            patches=padded_img[:,:,up:down+kh-1,t:t+kw].clone()
            
            model_output=model(rescaling(patches),False,up,down)
            means,coeffs,log_scales, pi=compute_stats(model_output.view(-1,bs).t())

            for c in range(0,3):    
                if c==0:
                    mean=means[:,0:1,:]
                elif c==1:
                    c_0=rescaling(sheared_o_img[0,0:1,up:down,t]/255.).t().unsqueeze(-1)
                    mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_0
                else:
                    c_1=rescaling(sheared_o_img[0,1:2,up:down,t]/255.).t().unsqueeze(-1)
                    mean=means[:,2:3, :] + coeffs[:,1:2, :]* c_0 +coeffs[:,2:3, :] * c_1

                cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                c_list.extend(np.take_along_axis(cdf_min_table,sheared_o_img[0,c,up:down,t].numpy().reshape(-1,1),axis=-1).reshape(-1))
                p_list.extend(np.take_along_axis(probs_table,sheared_o_img[0,c,up:down,t].numpy().reshape(-1,1),axis=-1).reshape(-1)) 
            padded_img[0,:,kh-1+up:kh-1+down,kw+t]=sheared_o_img[0,:,up:down,t]/255.

    ans_stack=ANSStack(s_prec = 32,t_prec = 16, p_prec=p_prec)              
    for i in np.arange(len(c_list)-1,-1,-1):
        c_min,p=c_list[i],p_list[i]
        ans_stack.push(c_min,p)
    return ans_stack                

    


    
def ans_decompression(model,ans_stack,Q,K,p_prec=16):
    model.eval()
    D,O,T,up_batch,down_batch,bs_batch=Q
    kh,kw=K
    p2d=[kw,0,kh-1,0]
    decode_img=shear(torch.zeros([1,3,D,D]),O)
    padded_img = F.pad(decode_img.clone(), p2d, "constant", 0)
    with torch.no_grad():      
        for t in range(0,T):
            up=up_batch[t]
            down=down_batch[t]
            bs=bs_batch[t]
            decoded_column=torch.zeros([3,bs])

            patches=padded_img[:,:,up:down+kh-1,t:t+kw].clone()
            model_output=model(rescaling(patches),False,up,down)
            means,coeffs,log_scales, pi=compute_stats(model_output.view(-1,bs).t())
            
            for c in range(0,3): 
                if c==0:
                    mean=means[:,0:1, :]
                elif c==1:
                    c_0=rescaling(decoded_column[0:1,:]/255.).t().unsqueeze(-1)
                    mean=means[:,1:2, :] + coeffs[:,0:1, :]* c_0
                else:
                    c_1=rescaling(decoded_column[1:2,:]/255.).t().unsqueeze(-1)
                    mean=means[:,2:3, :] + coeffs[:,1:2, :]* c_0 +coeffs[:,2:3, :] * c_1
                    
                cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                for h in range(0,bs):
                    s_bar = ans_stack.pop()
                    pt=np.searchsorted(cdf_min_table[h], s_bar, side='right', sorter=None)-1
                    decoded_column[c,h]=int(pt)
                    cdf,p=int(cdf_min_table[h][pt]),int(probs_table[h][pt])
                    ans_stack.update(s_bar,cdf,p)
                    
            padded_img[0,:,kh-1+up:kh-1+down,kw+t]=decoded_column/255.
            decode_img[0,:,up:down,t]=decoded_column
        return shear_inv(decode_img,O)[0]
                            