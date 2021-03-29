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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_audio_len=576000
    freeze_for_epochs=0
    transliterate=True
    language_identification=False #only For language identification task
    language_identification_asr=True # for both tasks simultaneously
    lang_param=0.25

    #inference for LM install Kenlm and make a 5-gram model
    lm_model=None
    asr_model='/content/gdrive/MyDrive/ASR_wav2vec_weights/LID_ASR/pytorch_model2.bin'
    file_path=""

def get_all_params_dict(config):
    params = {}
    for k, v in config.__dict__.items():
        if not ( callable(v) or (k.startswith('__') and k.endswith('__'))):
            params[k]=v
    return params
