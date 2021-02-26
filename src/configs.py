import os
import torch

class config:
    
    data_dir="/home/krishnarajule3/ASR/data/Hindi-English/"
    data_loading_script="/home/creativityinczenyoga/datasets/code_switch-asr"

    model="facebook/wav2vec2-base-960h"
    LR=1e-5
    clip_grad_norm=1.0
    EPOCHS=50
    num_iters_checkpoint=2000
    prev_checkpoint=""
    output_directory="./model/"
    
    os.makedirs(output_directory, exist_ok=True)
    
    BATCH_SIZE=5
    SHUFFLE=False
    eval=True
    train=True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_params_dict(config):
    params = {}
    for k, v in config.__dict__.items():
        if not ( callable(v) or (k.startswith('__') and k.endswith('__'))):
            params[k]=v
    return params
