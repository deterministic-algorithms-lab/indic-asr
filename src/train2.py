import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import soundfile as sf

import argparse
from tqdm import tqdm
from configs import config
from model import get_model
from tokenizer import Wav2Vec2Tok
from datasets import load_dataset


def find_lengths(logits, pad_id: int) -> torch.FloatTensor:
    """
    Function to find lengths of output sequences
    """
    preds = torch.argmax(logits, dim=-1)
    return torch.sum(torch.where(preds!=pad_id, 1, 0), axis=-1)

def save_checkpoint(model, name: str):
    print("saving model!")
    model.save_pretrained(os.path.join(config.output_directory, config.model+'_'+name))
    
def load_checkpoint(model, path: str):
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"))
    print("model loaded!")
    return model
    
def train_model(model, tokenizer, train_dataloader, val_dataloader):
    
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    ctc_loss = nn.CTCLoss()
    
    iters=0
    num_train_batches = len(train_dataloader)
    loss=0
    val_losses = []

    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        pbar=tqdm(train_dataloader, desc="Training epoch %d"%(epoch))
        
        for i, d in enumerate(pbar):
            pbar.set_postfix(loss =loss)
            
            iters+=1
            
            optimizer.zero_grad()

            input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(device)

            logits = model(input_values).logits

            labels, label_lengths = tokenizer.batch_tokenize(d['text'])

            loss = ctc_loss(logits.transpose(0,1), labels, 
                            find_lengths(logits, tokenizer.pad_token_id), label_lengths)

            # print("Training loss : ", loss)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), configs.clip_grad_norm)

            optimizer.step()

            loss = loss.item()
            epoch_loss += loss
            
            if(iters%config.num_iters_checkpoint==0):
                
                val_losses.append(eval_model(model, tokenizer, val_dataloader))
                if min(val_losses)==val_losses[-1]:
                    save_checkpoint(model, str(iters))
        
        print("Mean loss for epoch %d : "%epoch, (epoch_loss / num_train_batches))
    
    save_checkpoint(model, str(iters))
    
    
def eval_model(model, tokenizer, val_dataloader):
    model.eval()
    
    ctc_loss = nn.CTCLoss()

    epoch_loss = 0
    
    num_valid_batches = len(val_dataloader)

    pbar = tqdm(val_dataloader, desc="Validataion")
    
    for i, d  in enumerate(pbar):
        pbar.set_postfix(loss = loss)
        
        input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(device)

        logits = model(input_values).logits

        labels, label_lengths = tokenizer.batch_tokenize(d['text'])
        
        loss = ctc_loss(logits.transpose(0,1), labels, find_lengths(logits, tokenizer.pad_token_id), label_lengths)
        
        loss=loss.item()
        
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))
    return (epoch_loss / num_valid_batches)

if __name__ =='__main__':
    
    tokenizer = Wav2Vec2Tok.from_pretrained(config.model)
    
    model = get_model(tokenizer)

    if(config.prev_checkpoint!=""):
        model=load_checkpoint(model,config.prev_checkpoint)
    
    params = {'batch_size': config.BATCH_SIZE,
              'shuffle': config.SHUFFLE,}
    
    print("running on ", configs.device)

    train_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[10%:]", writer_batch_size=1000)
    val_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[:10%]", writer_batch_size=1000)
    test_dataset = load_dataset(config.data_loading_script, name=config.data_dir, split="test", writer_batch_size=1000)

    if(config.train):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, **params)
        train_model(model, tokenizer, train_dataloader, val_dataloader)
    
    if(config.eval):
        test_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, **params)
        eval_model(model, tokenizer, val_dataloader, device)
    
    print("TRAINING DONE!")
