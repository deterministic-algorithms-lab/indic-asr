import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from configs import config

def get_model(tokenizer):
    
    model = Wav2Vec2.from_pretrained(config.model)
    
    pt_wts = model.lm_head.weight
    pt_bias = model.lm_head.bias

    new_lm_head = nn.Linear(pt_wts.shape[1], len(tokenizer)+3)

    init_wts = new_lm_head.weight.clone().detach()
    init_bs = new_lm_head.bias.clone().detach()
    init_wts[:pt_wts.shape[0], :] = pt_wts.clone().detach()
    init_wts[pt_wts.shape[0]:, :] = torch.mean(pt_wts.clone().detach(), dim=0)
    init_bs[:pt_bias.shape[0]] = pt_bias.clone().detach()
    init_bs[pt_wts.shape[0]:] = torch.mean(pt_bias.clone().detach(), dim=0)

    with torch.no_grad():
        new_lm_head.weight = nn.Parameter(init_wts)
        new_lm_head.bias = nn.Parameter(init_bs)

    model.lm_head = new_lm_head
    
    return model.to(config.device)