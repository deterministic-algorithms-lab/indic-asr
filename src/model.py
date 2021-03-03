import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from configs import config

class Wav2Vec2(Wav2Vec2ForCTC):
    def __init__(self, conf):
        super().__init__(conf)

    def forward(
        self,
        input_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if config.cur_epoch<=config.freeze_for_epochs:
            
            with torch.no_grad():
                outputs = self.wav2vec2(
                    input_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            
            outputs = self.wav2vec2(
                    input_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return CausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def get_model(tokenizer):
    
    model = Wav2Vec2.from_pretrained(config.model)
    pt_wts = model.lm_head.weight
    pt_bias = model.lm_head.bias

    new_lm_head = nn.Linear(pt_wts.shape[1], len(tokenizer))

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