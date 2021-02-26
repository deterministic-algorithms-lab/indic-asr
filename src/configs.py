import os
import torch

class config:
    
    data_dir="/home/krishnarajule3/ASR/data/Hindi-English/
    data_loading_script="/home/datasets/code_switch_asr"

    model="facebook/wav2vec2-base-960h"
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