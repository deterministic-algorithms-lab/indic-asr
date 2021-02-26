import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from configs import config

def get_model(tokenizer):
    
    model = Wav2Vec2ForCTC.from_pretrained(config.model)
    pt_wts = model.lm_head.weight
    pt_bias = model.lm_head.bias

    new_lm_head = nn.Linear(pt_wts.shape[1], len(tokenizer))

    init_wts = new_lm_head.weight.clone().detach()
    init_bs = new_lm_head.bias.clone().detach()
    init_wts[:pt_wts.shape[0], :] = pt_wts.clone().detach()
    init_bs[:pt_bias.shape[0]] = pt_bias.clone().detach()

    with torch.no_grad():
        new_lm_head.weight = nn.Parameter(init_wts)
        new_lm_head.bias = nn.Parameter(init_bs)

    model.lm_head = new_lm_head
    
    return model.to(config.device)