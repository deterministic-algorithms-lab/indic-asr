import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from configs import config

class Wav2Vec2(Wav2Vec2ForCTC):
    def __init__(self, conf):
        super().__init__(conf)
        
#             if config.language_identification or config.language_identification_asr:
#                 self.lang_head=nn.Linear()
    def forward(
        self,
        input_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if config.cur_epoch<config.freeze_for_epochs:
            
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
        
        logits,logits2=None,None
        
        if not config.language_identification:
            logits1 = self.lm_head(hidden_states)
            
        if config.language_identification or config.language_identification_asr:
            logits2= self.lang_head(hidden_states)
        
        if not return_dict:
            if config.language_identification_asr:

                output1 = (logits1,) + outputs[1:]
                output2= (logits2,) + outputs[1:]
                return (output1,output2)
            if config.language_identification:
                output2= (logits2,) + outputs[1:]
                return output2
            
            output1= (logits1,) + outputs[1:]
            return output1
            
        
        if config.language_identification_asr:
            return CausalLMOutput(logits=logits1, hidden_states=outputs.hidden_states, attentions=outputs.attentions),CausalLMOutput(logits=logits2, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
                
        if config.language_identification:
            return CausalLMOutput(logits=logits2, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
                    
        return CausalLMOutput(logits=logits1, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def get_model(tokenizer):
    
    model = Wav2Vec2.from_pretrained(config.model)
    
    if not config.language_identification:
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
    
    if config.language_identification or config.language_identification_asr:
        model.lang_head=nn.Linear(model.lm_head.weight.shape[1],3)
    
    
    return model.to(config.device)