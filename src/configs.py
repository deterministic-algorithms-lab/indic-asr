import os
import torch

class config:
    
    data_dir="/home/krishnarajule3/ASR/data/Hindi-English/"
    data_loading_script="/home/datasets/code_switch_asr"

    use_monolingual=False
    monolingual_data_dir="/home/krishnarajule3/ASR/data/Hindi/"
    
    model="facebook/wav2vec2-base-960h"
    fast_LR=1e-3                                                                    #To be used when initial weights are frozen
    LR=1e-6
    clip_grad_norm=1.0
    EPOCHS=0
    num_iters_checkpoint=70
    prev_checkpoint=""
    output_directory="./model/"
    
    os.makedirs(output_directory, exist_ok=True)
    
    BATCH_SIZE=5
    SHUFFLE=False
    eval=False
    train=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_audio_len=576000
    freeze_for_epochs=0
    transliterate=False
    
    #Support for additional languages
    Telugu=(3072,3199+1)
    Tamil=(2944,3071+1)

    Oriya=(2816,2943+1)
    Gujrati=(2688,2815+1)
    Hindi=(2304,2431+1)
    Marathi=Hindi
    
    Language=Gujrati #select the language
    
    mono=True
    mono_train_path="./"
    mono_test_path="./"

def get_all_params_dict(config):
    params = {}
    for k, v in config.__dict__.items():
        if not ( callable(v) or (k.startswith('__') and k.endswith('__'))):
            params[k]=v
    return params
