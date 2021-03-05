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
import random

def find_lengths(logits, pad_id: int) -> torch.FloatTensor:
    """
    Function to find lengths of output sequences
    """
    #preds = torch.argmax(logits, dim=-1)
    #return torch.sum(torch.where(preds!=pad_id, 1, 0), axis=-1)
    return torch.tensor([logits.shape[1]]*logits.shape[0], dtype=torch.int16, device=config.device)

def save_checkpoint(model, name: str):
    print("saving model!")
    model_path = os.path.join(config.output_directory, config.model+'_'+name)
    model.save_pretrained(model_path)

def load_checkpoint(model, path: str):
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"))
    print("model loaded!")
    return model


def train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataset):
    
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.fast_LR if config.freeze_for_epochs>0 else config.LR)
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    
    iters=0
    num_train_batches = len(train_dataloader)
    loss=0
    val_losses = []

    for epoch in range(config.EPOCHS):
        
        config.cur_epoch = epoch
        if config.cur_epoch>config.freeze_for_epochs:
            optimizer = optim.Adam(model.parameters(), lr=config.LR)

        loss=0
        epoch_loss = 0
        pbar=tqdm(train_dataloader, desc="Training epoch %d"%(epoch))
        
        for i, d in enumerate(pbar):
            pbar.set_postfix(loss =loss)
            
            iters+=1
            
            input_values, labels, label_lengths = d
            if input_values.shape[1]>config.max_audio_len:
                print("skipping batch : ", i)
                continue
            
            optimizer.zero_grad()

            logits = model(input_values).logits

            loss = ctc_loss(logits.transpose(0,1), labels, 
                            find_lengths(logits, tokenizer.pad_token_id), label_lengths)

            # print("Training loss : ", loss)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            loss = loss.item()
            epoch_loss += loss
            
            if(iters%config.num_iters_checkpoint==0):
                model.eval()
                
                val_losses.append(eval_model(model, tokenizer, val_dataloader))
                
                wer_score = compute_metric(model, tokenizer, test_dataset)
                
                wandb.log({'validation_loss' : val_losses[-1],
                            'wer_on_test_set': wer_score})
                
                model.train()
                if min(val_losses)==val_losses[-1]:
                    save_checkpoint(model, str(epoch))
        
        print("Mean loss for epoch %d : "%epoch, (epoch_loss / num_train_batches))

    save_checkpoint(model, str(iters))

def eval_model(model, tokenizer, val_dataloader):
    
    ctc_loss = nn.CTCLoss(zero_infinity=True)

    loss=0
    epoch_loss = 0
    
    num_valid_batches = len(val_dataloader)

    pbar = tqdm(val_dataloader, desc="Validataion")
    
    for i, d  in enumerate(pbar):
        pbar.set_postfix(loss = loss)
        
        input_values, labels, label_lengths = d
        
        logits = model(input_values).logits
        
        loss = ctc_loss(logits.transpose(0,1), labels, find_lengths(logits, tokenizer.pad_token_id), label_lengths)
        
        loss = loss.item()
        
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))
    return (epoch_loss / num_valid_batches)

def compute_metric(model, tokenizer, test_dataset):
    metric = load_metric('wer')

    pbar = tqdm(test_dataset, desc="Computing metric")

    show_sample_no = random.randint(1, len(test_dataset)-1)

    for i, d in enumerate(pbar):
        
        input_values = tokenizer(d["speech"], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)

        logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1).cpu()
        transcriptions = tokenizer.batch_decode(predicted_ids)
        transcriptions = tokenizer.revert_transliteration(transcriptions)
        
        reference = d['text'].upper() 
        
        if i==show_sample_no or i==0:
            print("Sample prediction: ", transcriptions[0])
            print("Sample reference: ", reference)
        
        metric.add_batch(predictions=transcriptions, 
                         references=[reference])
    
    score = metric.compute()
    print("Evaluation metric: ", score)
    return score

def collate_fn(batch, tokenizer):
    speech_lis = [elem["speech"] for elem in batch]
    text_lis = [elem["text"].upper() for elem in batch]
    
    input_values = tokenizer(speech_lis, return_tensors="pt", 
                                     padding='longest').input_values

    labels, label_lengths = tokenizer.batch_tokenize(text_lis)

    return (input_values.to(config.device), labels.to(config.device), label_lengths.to(config.device))

if __name__ =='__main__':
    all_params_dict = get_all_params_dict(config)
    
    wandb.init(project="wav2vec2.0", entity="interspeech-asr", config=all_params_dict)

    tokenizer = Wav2Vec2Tok.from_pretrained(config.model)
    
    model = get_model(tokenizer)
    
    wandb.watch(model)
    config.output_directory = wandb.run.dir

    if(config.prev_checkpoint!=""):
        model=load_checkpoint(model,config.prev_checkpoint)
    
    params = {'batch_size': config.BATCH_SIZE,}
    
    print("running on ", config.device)

    train_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[2%:]", writer_batch_size=1000)
    val_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[:2%]", writer_batch_size=1000)
    test_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="test", writer_batch_size=1000)

    if(config.train):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataset)
    
    if(config.eval):
        print(compute_metric(model, tokenizer, test_dataset))
    
    print("TRAINING DONE!")
