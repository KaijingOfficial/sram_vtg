
import torch
from torch import nn
import torch.nn.functional as F


# evidence predictor
class CustomNormalInvGamma(nn.Module):
    def __init__(self, span_embed_model, der_model, out_units=2):
        super().__init__()
        self.conv = span_embed_model
        self.dense = der_model
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        mu = self.conv(x) # [64,75,512]-->[64,75,2]
        out = self.dense(x) # [64,75,512]-->[64,75,6]
        logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)

        mu = mu.sigmoid()
        idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).to(mu.device)
        idx_mask = idx_mask.repeat(mu.shape[0], mu.shape[1], 1)
        mu = mu * idx_mask
        
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
def nig_nll(dist_params, mask):

    gamma = dist_params[0]
    v = dist_params[2]
    alpha = dist_params[3]
    beta = dist_params[4]
    y = dist_params[1]

    two_beta_lambda = 2 * beta * (1 + v) 
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    tm = F.softplus((v * (y - gamma) ** 2 + two_beta_lambda))
    t3 = (alpha + 0.5) * tm.log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    masked_nll = nll * mask
    return masked_nll.sum() / mask.sum()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
def nig_reg(dist_params, mask):

    gamma = dist_params[0]
    v = dist_params[2]
    alpha = dist_params[3]
    y = dist_params[1]

    mis_ = (y - gamma).abs()
    mis = mis_.detach() # we propose to detach here to block gradient from edl loss to main backbone, and it really work (otherwise the performance can degrade)
    reg = (mis) * (2 * v + alpha)
    masked_reg = reg * mask
    return masked_reg.sum() / mask.sum()


# Our proposed regularization
def nig_geom(dist_params, mask, norm_mode='maxmin_batch', cons_mode='exp_2-cir', coef_list=[1,1,1]):

    src_spans = dist_params[0]
    gt_spans = dist_params[1]
    delta_spans = abs(src_spans - gt_spans)
    au = dist_params[5]
    eu = dist_params[6]
    evi = dist_params[7]

    coef_au_acc, coef_eu_acc, coef_evi_mis = coef_list[0], coef_list[1], coef_list[2]

    if norm_mode == 'maxmin_batch':
        b, n, d = delta_spans.shape

        ################## max-min 归一化 ###################
        # 归一化mis/acc
        max = torch.max(delta_spans.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(delta_spans.flatten(), dim=0)[0] # [b,75,2]
        length = max - min # [b,75,2]
        mis = (delta_spans - min) / length # [b,75,2]
        # 归一化au
        max = torch.max(au.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(au.flatten(), dim=0)[0] # [b,75,2]
        length = max - min # [b,75,2]
        au = (au - min) / length # [b,75,2]
        # 归一化eu
        max = torch.max(eu.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(eu.flatten(), dim=0)[0] # [b,75,2]
        length = max - min # [b,75,2]
        eu = (eu - min) / length # [b,75,2]
        # 归一化evi
        max = torch.max(evi.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(evi.flatten(), dim=0)[0] # [b,75,2]
        length = max - min # [b,75,2]
        evi = (evi - min) / length # [b,75,2]

    if norm_mode == 'maxmin_video':
        b, n, d = delta_spans.shape

        ################## max-min 归一化 ###################
        # 归一化mis/acc
        max = torch.repeat_interleave(torch.max(delta_spans, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        min = torch.repeat_interleave(torch.min(delta_spans, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        length = max - min # [b,75,2]
        mis = (delta_spans - min) / length # [b,75,2]
        # 归一化au
        max = torch.repeat_interleave(torch.max(au, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        min = torch.repeat_interleave(torch.min(au, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        length = max - min # [b,75,2]
        au = (au - min) / length # [b,75,2]
        # 归一化eu
        max = torch.repeat_interleave(torch.max(eu, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        min = torch.repeat_interleave(torch.min(eu, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        length = max - min # [b,75,2]
        eu = (eu - min) / length # [b,75,2]
        # 归一化evi
        max = torch.repeat_interleave(torch.max(evi, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        min = torch.repeat_interleave(torch.min(evi, dim=1, keepdims=True)[0], repeats=n, dim=1) # [b,75,2]
        length = max - min # [b,75,2]
        evi = (evi - min) / length # [b,75,2]

    if norm_mode == 'activate':
        b, n, d = delta_spans.shape

        ################## max-min 归一化 ###################
        # 归一化mis/acc
        max = torch.max(delta_spans.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(delta_spans.flatten(), dim=0)[0] # [b,75,2]
        mis = delta_spans - min # [b,75,2]
        mis = F.tanh(mis)
        # 归一化au
        max = torch.max(au.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(au.flatten(), dim=0)[0] # [b,75,2]
        au = au - min # [b,75,2]
        au = F.tanh(au)
        # 归一化eu
        max = torch.max(eu.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(eu.flatten(), dim=0)[0] # [b,75,2]
        eu = eu - min # [b,75,2]
        eu = F.tanh(eu)
        # 归一化evi
        max = torch.max(evi.flatten(), dim=0)[0] # [b,75,2]
        min = torch.min(evi.flatten(), dim=0)[0] # [b,75,2]
        evi = evi - min # [b,75,2]
        evi = F.tanh(evi)

    ################## detach ###################
    mis_ = mis.detach()
    acc = 1 - mis_

    if cons_mode == 'exp_1-line':
        ################# 1-line ###############
        # loss: au + acc = 1
        geom_au_acc = torch.exp(coef_au_acc * (acc + au - 1) ** 2)
        # loss: eu + acc = 1
        geom_eu_acc = torch.exp(coef_eu_acc * (acc + eu - 1) ** 2)
        # loss: evi + mis = 1
        geom_evi_mis = torch.exp(coef_evi_mis * (mis_ + evi - 1) ** 2)

    if cons_mode == 'lin_1-line':
        ################# 1-line ###############
        # loss: au + acc = 1
        geom_au_acc = coef_au_acc * (acc + au - 1) ** 2
        # loss: eu + acc = 1
        geom_eu_acc = coef_eu_acc * (acc + eu - 1) ** 2
        # loss: evi + mis = 1
        geom_evi_mis = coef_evi_mis * (mis_ + evi - 1) ** 2

    if cons_mode == 'exp_2-line':
        ################# 2-line ###############
        # loss: au + acc = 1 / au = acc
        geom_au_acc1 = torch.exp(coef_au_acc * (acc + au - 1) ** 2)
        geom_au_acc2 = torch.exp(coef_au_acc * (acc - au) ** 2)
        geom_au_acc = geom_au_acc1 - geom_au_acc2
        # loss: eu + acc = 1 / eu = acc
        geom_eu_acc1 = torch.exp(coef_eu_acc * (acc + eu - 1) ** 2)
        geom_eu_acc2 = torch.exp(coef_eu_acc * (acc - eu) ** 2)
        geom_eu_acc = geom_eu_acc1 - geom_eu_acc2
        # loss: evi + mis = 1 / evi = mis
        geom_evi_mis1 = torch.exp(coef_evi_mis * (mis_ + evi - 1) ** 2)
        geom_evi_mis2 = torch.exp(coef_evi_mis * (mis_ - evi) ** 2)
        geom_evi_mis = geom_evi_mis1 - geom_evi_mis2

    if cons_mode == 'lin_2-line':
        ################# 2-line ###############
        # loss: au + acc = 1 / au = acc
        geom_au_acc1 = coef_au_acc * (acc + au - 1) ** 2
        geom_au_acc2 = coef_au_acc * (acc - au) ** 2
        geom_au_acc = geom_au_acc1 - geom_au_acc2
        # loss: eu + acc = 1 / eu = acc
        geom_eu_acc1 = coef_eu_acc * (acc + eu - 1) ** 2
        geom_eu_acc2 = coef_eu_acc * (acc - eu) ** 2
        geom_eu_acc = geom_eu_acc1 - geom_eu_acc2
        # loss: evi + mis = 1 / evi = mis
        geom_evi_mis1 = coef_evi_mis * (mis_ + evi - 1) ** 2
        geom_evi_mis2 = coef_evi_mis * (mis_ - evi) ** 2
        geom_evi_mis = geom_evi_mis1 - geom_evi_mis2

    if cons_mode == 'exp_2-cir':
        ################# 2-circle ###############
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(au,acc)
        geom_au_acc1 = torch.exp(coef_au_acc * (acc ** 2 + (au - 1) ** 2))
        geom_au_acc2 = torch.exp(coef_au_acc * (au ** 2 + (acc - 1) ** 2))
        geom_au_acc = geom_au_acc1 - geom_au_acc2
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(eu,acc)
        geom_eu_acc1 = torch.exp(coef_eu_acc * (acc ** 2 + (eu - 1) ** 2))
        geom_eu_acc2 = torch.exp(coef_eu_acc * (eu ** 2 + (acc - 1) ** 2))
        geom_eu_acc = geom_eu_acc1 - geom_eu_acc2
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(evi,mis)
        geom_evi_mis1 = torch.exp(coef_evi_mis * (mis ** 2 + (evi - 1) ** 2))
        geom_evi_mis2 = torch.exp(coef_evi_mis * (evi ** 2 + (mis - 1) ** 2))
        geom_evi_mis = geom_evi_mis1 - geom_evi_mis2

    if cons_mode == 'lin_2-cir':
        ################# 2-circle ###############
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(au,acc)
        geom_au_acc1 = coef_au_acc * (acc ** 2 + (au - 1) ** 2)
        geom_au_acc2 = coef_au_acc * (au ** 2 + (acc - 1) ** 2)
        geom_au_acc = geom_au_acc1 - geom_au_acc2
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(eu,acc)
        geom_eu_acc1 = coef_eu_acc * (acc ** 2 + (eu - 1) ** 2)
        geom_eu_acc2 = coef_eu_acc * (eu ** 2 + (acc - 1) ** 2)
        geom_eu_acc = geom_eu_acc1 - geom_eu_acc2
        # loss: d[p,(1,0)]=1 / d[p,(0,1)]=1 ; p=(evi,mis)
        geom_evi_mis1 = coef_evi_mis * (mis ** 2 + (evi - 1) ** 2)
        geom_evi_mis2 = coef_evi_mis * (evi ** 2 + (mis - 1) ** 2)
        geom_evi_mis = geom_evi_mis1 - geom_evi_mis2

    # total loss
    geom = geom_au_acc + geom_eu_acc + geom_evi_mis

    masked_geom = geom * mask
    return masked_geom.sum() / mask.sum()
