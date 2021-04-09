import os
import torch

class config:
    
#     'facebook/wav2vec2-large-xlsr-53'
#     'facebook/wav2vec2-base-960h'
    model="facebook/wav2vec2-base-960h"
    fast_LR=1e-3              #To be used when initial weights are frozen
    LR=1e-5
    clip_grad_norm=1.0
    EPOCHS=1000
    num_iters_checkpoint=56000
    prev_checkpoint=""
    
    output_directory="./model/"
    
    os.makedirs(output_directory, exist_ok=True)
    
    BATCH_SIZE=8
    SHUFFLE=True
    eval=True
    train=True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_audio_len=576000
    freeze_for_epochs=0
    transliterate=False
    cur_epoch=0

    #Support for additional languages
    Telugu=(3072,3199+1)
    Tamil=(2944,3071+1)

    Odia=(2816,2943+1)
    Gujarati=(2688,2815+1)
    Hindi=(2304,2431+1)
    Bengali=(2433,2554+1)
    Marathi=Hindi
    
    Language=[Hindi,Gujarati,Telugu,Tamil,Odia]         #select the language (can add multiple languages to the list)
    
    #Mono-Language Training
    
    mono=True                     #to specify training for the monolingual language (to use mono dataset)

    mono_train_path=["/home/krishnarajule3/ASR/data/Hindi/train","/home/krishnarajule3/ASR/data/Marathi/train","/home/krishnarajule3/ASR/data/Odia/train",
			"/home/krishnarajule3/ASR/data/Gujarati/gu-in-Train","/home/krishnarajule3/ASR/data/Tamil/ta-in-Train","/home/krishnarajule3/ASR/data/Telegu/te-in-Train"
			]       #path to training folder 

    mono_test_path=["/home/krishnarajule3/ASR/data/Hindi/test","/home/krishnarajule3/ASR/data/Marathi/test","/home/krishnarajule3/ASR/data/Odia/test",
			"/home/krishnarajule3/ASR/data/Gujarati/gu-in-Test","/home/krishnarajule3/ASR/data/Tamil/ta-in-Test","/home/krishnarajule3/ASR/data/Telegu/te-in-Train"
			]        #path to testing folder
    
    #Code Switched Training (set mono=False, to use code-switched loader.py)
    
    data_dir="/home/krishnarajule3/ASR/data/Hindi-English/"
    data_loading_script="/home/datasets/code_switch_asr"

    use_monolingual=False
    monolingual_data_dir="/home/krishnarajule3/ASR/data/Hindi/"
    
def get_all_params_dict(config):
    params = {}
    for k, v in config.__dict__.items():
        if not ( callable(v) or (k.startswith('__') and k.endswith('__'))):
            params[k]=v
    return params



