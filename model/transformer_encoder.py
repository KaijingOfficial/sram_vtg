import copy
import pdb
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sys

class XConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self,
                 x_layers=4,):
        # self.l_layers = l_layers
        self.x_layers = x_layers
        # self.r_layers = r_layers
        self.hidden_size = 512
        self.visual_feat_dim = 512
        self.intermediate_size = 512
        self.num_attention_heads = 8
        self.attention_probs_dropout_prob=0
        self.hidden_dropout_prob=0
        self.hidden_act = 'gelu'
        # self.cache_attention = True


    def set_dims(self, feat_dim, hidden_size,num_cross_layers):
        self.visual_feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.x_layers = num_cross_layers

XENCODER_CONFIG = XConfig()

def swish(x):
    return x * torch.sigmoid(x)

BertLayerNorm = torch.nn.LayerNorm
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        self.output = BertAttOutput(config)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, input_tensor, ctx_tensor, input_pos, ctx_pos, input_attn_mask, ctx_att_mask):
        
        q = self.with_pos_embed(input_tensor, input_pos)
        k = self.with_pos_embed(ctx_tensor, ctx_pos)
        v = ctx_tensor
        
        output, attention_map = self.att(q,k,v, key_padding_mask=ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_map

class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        self.output = BertAttOutput(config)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, input_tensor, input_pos, attention_mask):
        q = k =  self.with_pos_embed(input_tensor, input_pos)
        v = input_tensor       
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(q,k,v,key_padding_mask=attention_mask)[0]
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FlipXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        

    def cross_att(self, lang_input, lang_attention_mask, lang_pos, visn_input, visn_attention_mask, visn_pos):
        
        # Cross Attention
        lang_att_output, t2v_attention_score = self.visual_attention(lang_input, visn_input, 
                                                lang_pos, visn_pos, 
                                                input_attn_mask=lang_attention_mask, 
                                                ctx_att_mask=visn_attention_mask)
        
        visn_att_output, v2t_attention_score= self.visual_attention(visn_input, lang_input, 
                                                visn_pos, lang_pos, 
                                                input_attn_mask=visn_attention_mask, 
                                                ctx_att_mask=lang_attention_mask)
        
        return lang_att_output, visn_att_output,t2v_attention_score,v2t_attention_score

    def self_att(self, lang_input, lang_attention_mask, lang_pos, visn_input, visn_attention_mask, visn_pos):
        
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_pos, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_pos, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,lang_pos,
                      visn_feats, visn_attention_mask,vis_pos):
        
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        
        lang_att_output, visn_att_output,t2v_attention_score,v2t_attention_score = \
            self.cross_att(lang_att_output, lang_attention_mask,lang_pos,
                                                          visn_att_output, visn_attention_mask,vis_pos)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,lang_pos,
                                                         visn_att_output, visn_attention_mask,vis_pos)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output, t2v_attention_score, v2t_attention_score

class XEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        # self.visn_fc = VisualFeatEncoder(config)

        # # Number of layers
        # self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = config.x_layers

        # self.num_r_layers = VISUAL_CONFIG.r_layers
        # print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
        #       (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        # self.layer = nn.ModuleList(
        #     [BertLayer(config) for _ in range(self.num_l_layers)]
        # )
        self.x_layers = nn.ModuleList(
            [FlipXLayer(config) for _ in range(self.num_x_layers)]
        )
        # self.r_layers = nn.ModuleList(
        #     [BertLayer(config) for _ in range(self.num_r_layers)]
        # )

    def forward(self, lang_feats, lang_attention_mask,lang_pos,
                visn_feats, visn_attention_mask,visn_pos):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        # visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        # for layer_module in self.layer:
        #     lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        # for layer_module in self.r_layers:
        #     visn_feats = layer_module(visn_feats, visn_attention_mask)
        attention_cache = {}
        # Run cross-modality layers
        for i,layer_module in enumerate(self.x_layers):
            lang_feats, visn_feats, t2v_attention_score, v2t_attention_score = layer_module(lang_feats, lang_attention_mask,lang_pos,
                                                  visn_feats, visn_attention_mask,visn_pos)
            attention_cache[f't2v_attn_layer_{str(i)}'] = t2v_attention_score
            attention_cache[f'v2t_attn_layer_{str(i)}'] = v2t_attention_score

        return lang_feats, visn_feats, attention_cache

class Transformer(nn.Module):

    def __init__(self, config=XENCODER_CONFIG):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.x_encoder = XEncoder(config)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, vidlen, txtlen):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d) -> my imple (batch_size, d) and #queries=1
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """

        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)


        visn_feats = src[:vidlen]
        visn_attention_mask = mask[:, :vidlen]
        visn_pos = pos_embed[:vidlen]
        lang_feats = src[-txtlen:,...]
        lang_attention_mask = mask[:, -txtlen:]
        lang_pos = pos_embed[-txtlen:,...]
        
        txt_memory, vid_memory, cached_attention = self.x_encoder(lang_feats,lang_attention_mask,lang_pos,
                                                visn_feats,visn_attention_mask,visn_pos)

        memory = torch.cat([vid_memory,txt_memory],dim=0)

        memory = memory.transpose(0, 1)

        return memory, cached_attention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    
    def build_config(args):
        config=XENCODER_CONFIG
        XENCODER_CONFIG.set_dims(feat_dim=args.hidden_dim, 
                                 hidden_size=args.hidden_dim,
                                 num_cross_layers=args.num_cross_layers)
        return config
    
    return Transformer(
        config=build_config(args)
    )

def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask

    return x


class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res = drop_path(x, self.drop_prob, self.training)
        return res.permute(1, 0, 2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")