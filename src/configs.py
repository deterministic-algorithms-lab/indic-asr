import os

class config:
    
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