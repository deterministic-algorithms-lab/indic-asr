import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import soundfile as sf

import argparse
from tqdm import tqdm
from configs import config, get_all_params_dict
from model import get_model
from tokenizer import Wav2Vec2Tok
from datasets import load_dataset, load_metric
import wandb

def find_lengths(logits, pad_id: int) -> torch.FloatTensor:
    """
    Function to find lengths of output sequences
    """
    preds = torch.argmax(logits, dim=-1)
    return torch.sum(torch.where(preds!=pad_id, 1, 0), axis=-1)

def save_checkpoint(model, name: str):
    print("saving model!")
    model_path = os.path.join(config.output_directory, config.model+'_'+name)
    model.save_pretrained(model_path)
    wandb.save(model_path)

def load_checkpoint(model, path: str):
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"))
    print("model loaded!")
    return model


def train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader):
    
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    ctc_loss = nn.CTCLoss()
    
    iters=0
    num_train_batches = len(train_dataloader)
    loss=0
    val_losses = []

    for epoch in range(config.EPOCHS):
        
        loss=0
        epoch_loss = 0
        pbar=tqdm(train_dataloader, desc="Training epoch %d"%(epoch))
        
        for i, d in enumerate(pbar):
            pbar.set_postfix(loss =loss)
            
            iters+=1
            
            optimizer.zero_grad()

            input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)

            logits = model(input_values).logits

            labels, label_lengths = tokenizer.batch_tokenize(d['text'])

            loss = ctc_loss(logits.transpose(0,1), labels, 
                            find_lengths(logits, tokenizer.pad_token_id), label_lengths)

            if i<100:
                print("Training loss : ", loss)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            loss = loss.item()
            epoch_loss += loss
            
            if(iters%config.num_iters_checkpoint==0):
                model.eval()
                
                val_losses.append(eval_model(model, tokenizer, val_dataloader))
                
                wandb.log({'validation_loss' : val_losses[-1],
                            'wer_on_test_set': compute_metric(model, tokenizer, test_dataloader)})
                
                model.train()
                if min(val_losses)==val_losses[-1]:
                    save_checkpoint(model, str(iters))
        
        print("Mean loss for epoch %d : "%epoch, (epoch_loss / num_train_batches))

    save_checkpoint(model, str(iters))

def eval_model(model, tokenizer, val_dataloader):
    
    ctc_loss = nn.CTCLoss()

    loss=0
    epoch_loss = 0
    
    num_valid_batches = len(val_dataloader)

    pbar = tqdm(val_dataloader, desc="Validataion")
    
    for i, d  in enumerate(pbar):
        pbar.set_postfix(loss = loss)
        
        input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)

        logits = model(input_values).logits

        labels, label_lengths = tokenizer.batch_tokenize(d['text'])
        
        loss = ctc_loss(logits.transpose(0,1), labels, find_lengths(logits, tokenizer.pad_token_id), label_lengths)
        
        loss = loss.item()
        
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))
    return (epoch_loss / num_valid_batches)

def compute_metric(model, tokenizer, test_dataloader):
    metric = load_metric('wer')

    pbar = tqdm(test_dataloader, desc="Computing metric")

    for i, d in enumerate(pbar):
        
        input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)

        logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = tokenizer.batch_decode(predicted_ids)
        
        metric.add_batch(predictions=transcriptions, references=d['text'])
    
    score = metric.compute()
    print("Evaluation metric: ", score)

if __name__ =='__main__':
    all_params_dict = get_all_params_dict(config)
    
    wandb.init(project="wav2vec2", entity="interspeech-asr", config=all_params_dict)

    tokenizer = Wav2Vec2Tok.from_pretrained(config.model)
    
    model = get_model(tokenizer)
    
    wandb.watch(model)
    
    if(config.prev_checkpoint!=""):
        model=load_checkpoint(model,config.prev_checkpoint)
    
    params = {'batch_size': config.BATCH_SIZE,
              'shuffle': config.SHUFFLE,}
    
    print("running on ", config.device)

    train_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[2%:]", writer_batch_size=1000)
    val_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[:2%]", writer_batch_size=1000)
    test_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="test", writer_batch_size=1000)

    if(config.train):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **params)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, **params)
        test_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, **params)
        train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader)
    
    if(config.eval):
        test_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, **params)
        eval_model(model, tokenizer, val_dataloader)
    
    print("TRAINING DONE!")
