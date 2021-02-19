import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import soundfile as sf

from tqdm import tqdm
from configs import config
from model import get_model
from tokenizer import Wav2Vec2Tok
from dataset import ASR
from datasets import load_dataset


def find_lengths(logits, pad_id: int) -> torch.FloatTensor:
    """
    Function to find lengths of output sequences
    """
    preds = torch.argmax(logits, dim=-1)
    return torch.sum(torch.where(preds!=pad_id, 1, 0), axis=-1)

def save_checkpoint(model,name):
    print("saving model!")
    model.save_pretrained(os.path.join(config.output_directory,config.model+'_'+name))
    
def load_checkpoint(model,path):
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"))
    print("model loaded!")
    return model
    
def train_model(model,tokenizer,train_dataloader,device):
    
    model.train()
    model.to(device)
    epoch_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    num_train_batches = len(train_dataloader)
    iters=0
    ctc_loss = nn.CTCLoss()
    loss=0
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        pbar=tqdm(train_dataloader,desc="Training epoch %d"%(epoch))
        for i, d in enumerate(pbar):
            # print(d)
            pbar.set_postfix(loss =loss)
            iters+=1
            optimizer.zero_grad()


            input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(device)

            logits = model(input_values).logits

            labels, label_lengths = tokenizer.batch_tokenize(d['text'])
            labels.to(device)
            label_lengths.to(device)    
            loss = ctc_loss(logits.transpose(0,1), labels, find_lengths(logits, tokenizer.pad_token_id), label_lengths)

            # print("Training loss : ", loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            loss=loss.item()
            epoch_loss += loss
        
            if(iters%config.num_iters_checkpoint==0):
                save_checkpoint(model,str(iters))
        
        print("Mean epoch %d loss: "%epoch, (epoch_loss / num_train_batches))
    
    save_checkpoint(model,str(iters))
    
    
def eval_model(model,tokenizer,val_dataloader,device):
    model.eval()
    model.to(device)
    epoch_loss = 0
    
    num_valid_batches = len(val_dataloader)
    ctc_loss=nn.CTCLoss()
    loss=0
    pbar=tqdm(val_dataloader, desc="Validataion")
    for i, d  in enumerate(pbar):
        pbar.set_postfix(loss =loss)
        input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(device)

        logits = model(input_values).logits

        labels, label_lengths = tokenizer.batch_tokenize(d['text'])
        labels.to(device)
        label_lengths.to(device)
        loss = ctc_loss(logits.transpose(0,1), labels, find_lengths(logits, tokenizer.pad_token_id), label_lengths)

        # loss=loss.item()
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))

    
    
    

if __name__ =='__main__':
    
    tokenizer=Wav2Vec2Tok.from_pretrained(config.model)
    
    model=get_model(tokenizer)

    if(config.prev_checkpoint!=""):
        model=load_checkpoint(model,config.prev_checkpoint)
    
    params = {'batch_size': config.BATCH_SIZE,
      'shuffle': config.SHUFFLE,}
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("running on ",device)
    
#     train_dataset,val_dataset=ASR()._split_generators()

    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds=ds.map(map_to_array)
    train_dataset,val_dataset=ds,ds

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)
    train_model(model,tokenizer,train_dataloader,device)
    
    if(config.eval):
        val_dataloader=torch.utils.data.DataLoader(dataset=val_dataset, **params)
        eval_model(model,tokenizer,val_dataloader,device)
    
    print("TRAINING DONE!")
    
    
    
    model = model.cpu()

    input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    print(transcription)

  
    
    
    
    
    #tqdm display loss
    