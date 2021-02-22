import os

class config:
    
    data_loader="data_loader.py"
    train_trans="/home/krishnarajule3/ASR/data/Hindi-English/train/transcript/text"
    train_audio="/home/krishnarajule3/ASR/data/Hindi-English/train/audio/"
    dev_trans="/home/krishnarajule3/ASR/data/Hindi-English/dev/transcript/text"
    train_audio="/home/krishnarajule3/ASR/data/Hindi-English/dev/audio/"
    test_trans="/home/krishnarajule3/ASR/data/Hindi-English/test/transcript/text"
    train_audio="/home/krishnarajule3/ASR/data/Hindi-English/test/audio/"
    
    model="facebook/wav2vec2-base-960h"
    LR=1e-6
    EPOCHS=0
    num_iters_checkpoint=70
    prev_checkpoint=""
    output_directory="./model/"
    
    os.makedirs(output_directory,exist_ok=True)
    
    BATCH_SIZE=1
    SHUFFLE=False
    eval=False
    train=False
