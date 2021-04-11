import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from configs import config

def get_model(tokenizer, n_langs=2):
    """Constructs the model with asr and language identification,
    from the base Wav2Vec2 model by modifying the last lm_head layer.
    Args:
        tokenizer: The tokenizer whose length is all the alphabets that 
                   the model can predict.
        n_langs: The number of different languages the model needs to distinguish between.
    Returns:
        The constructed model, having len(tokenizer)+n_langs+1 outputs in the last layer.
    """
    model = Wav2Vec2ForCTC.from_pretrained(config.model)
    
    pt_wts = model.lm_head.weight
    pt_bias = model.lm_head.bias

    new_lm_head = nn.Linear(pt_wts.shape[1], len(tokenizer)+(0 if n_langs<=1 else n_langs+1))

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