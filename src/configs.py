import os
import torch

class config:
    
    data_dir="/home/krishnarajule3/ASR/data/Hindi-English/"
    data_loading_script="./loader.py"

    use_monolingual=False
    monolingual_data_dir="/home/krishnarajule3/ASR/data/Hindi/"
    
    model="facebook/wav2vec2-base-960h"
    fast_LR=1e-3                                                                    #To be used when initial weights are frozen
    LR=1e-5
    clip_grad_norm=1.0
    EPOCHS=100
    num_iters_checkpoint=50000
    prev_checkpoint="./model/wav2vec2-base-960h_17/"
    output_directory="./model/"
    cur_epoch=0
    os.makedirs(output_directory, exist_ok=True)
    
    BATCH_SIZE=1
    SHUFFLE=False
    eval=True
    train=True
    load_from_disk=False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_audio_len=576000
    freeze_for_epochs=0
    
    transliterate=True

    #Hyperparameters for adjusting signal from language identification and asr
    lang_param=0.25
    asr_param=1.0

def get_all_params_dict(config):
    params = {}
    for k, v in config.__dict__.items():
        if not ( callable(v) or (k.startswith('__') and k.endswith('__'))):
            params[k]=v
    return params
