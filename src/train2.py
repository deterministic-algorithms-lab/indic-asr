import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import itertools
import soundfile as sf

import argparse
from tqdm import tqdm
from configs import config, get_all_params_dict
from model import get_model
from tokenizer import Wav2Vec2Tok
from datasets import load_dataset, load_metric, load_from_disk
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
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"),strict=False)
    print("model loaded!")
    return model

def get_datasets():
    """Returns train, valid, and test and mono datasets"""

    if config.load_from_disk:
        ds = load_from_disk(config.data_dir)
        data_dic = ds['train'].train_test_split(test_size=0.02)
        mono_dataset = load_from_disk(config.monolingual_data_dir) if config.use_monolingual else None
        return data_dic['train'], data_dic['test'], ds['test'], mono_dataset
    
    train_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[2%:]", writer_batch_size=1000)
    val_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[:2%]", writer_batch_size=1000)
    test_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="test", writer_batch_size=1000)
    
    if config.use_monolingual:
        mono_dataset = load_dataset(config.data_loading_script, data_dir=config.monolingual_data_dir, split="train", writer_batch_size=1000)
    else:
        mono_dataset = None
    
    return train_dataset, val_dataset, test_dataset, mono_dataset

def train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, mono_dataloader=None):
    
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.fast_LR if config.freeze_for_epochs>0 else config.LR)
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    
    iters=0
    num_train_batches = len(train_dataloader)
    loss=0
    wer_score = []

    for epoch in range(config.EPOCHS):
        
        config.cur_epoch = epoch
        if config.cur_epoch>=config.freeze_for_epochs:
            optimizer = optim.Adam(model.parameters(), lr=config.LR)

        loss = 0 
        epoch_loss = 0
        
        if mono_dataloader is not None:
            pbar=tqdm(itertools.chain(train_dataloader, mono_dataloader), desc="Training epoch %d"%(epoch))
        else:
            pbar=tqdm(train_dataloader, desc="Training epoch %d"%(epoch))
        
        for i, d in enumerate(pbar):
            pbar.set_postfix(loss=loss)
            
            iters+=1
            
            input_values, token_ids, token_seq_lengths, lang_ids, lang_labels_lengths = d
            
            if input_values.shape[1]>config.max_audio_len:
                print("skipping batch : ", i)
                continue
            
            optimizer.zero_grad()
            
            logits = model(input_values)
            token_logits, lang_logits = logits[...,:-3], logits[..., -3:]
            
            asr_loss = ctc_loss(token_logits.transpose(0,1), token_ids, 
                                find_lengths(token_logits, tokenizer.pad_token_id), token_seq_lengths)
            
            lang_class_loss = ctc_loss(lang_logits.transpose(0,1), lang_ids, 
                                       find_lengths(lang_logits, tokenizer.pad_token_id), lang_labels_lengths)
            
            loss = config.asr_param*asr_loss + config.lang_param*lang_class_loss
            
            #print("Training loss : ", loss)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            loss = loss.item()
            epoch_loss += loss
            
            if(iters%config.num_iters_checkpoint==0):
                model.eval()
                
                val_losses = eval_model(model, tokenizer, val_dataloader)
                
                wer_score.append(compute_metric(model, tokenizer, test_dataloader))
                
                wandb.log({'validation_loss' : val_losses,
                           'wer_on_test_set': wer_score[-1]})
                
                model.train()
                if min(wer_score)==wer_score[-1]:
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
        
        input_values, token_ids, token_seq_lengths, lang_ids, lang_labels_lengths = d
            
        if input_values.shape[1]>config.max_audio_len:
            print("skipping batch : ", i)
            continue
            
            
        logits = model(input_values)
        token_logits, lang_logits = logits[...,:-3], logits[..., -3:]
            
        asr_loss = ctc_loss(token_logits.transpose(0,1), token_ids, 
                            find_lengths(token_logits, tokenizer.pad_token_id), token_seq_lengths)
            
        lang_class_loss = ctc_loss(lang_logits.transpose(0,1), lang_ids, 
                                   find_lengths(lang_logits, tokenizer.pad_token_id), lang_labels_lengths)
            
        loss = config.asr_param*asr_loss + config.lang_param*lang_class_loss
            
        loss = loss.item()
        
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))
    return (epoch_loss / num_valid_batches)

def compute_metric(model, tokenizer, test_dataloader):
    
    wer_metric = load_metric('wer')
    acc_metric = load_metric('accuracy')

    pbar = tqdm(test_dataloader, desc="Computing metric")

    show_sample_no = random.randint(1, len(test_dataset)-1)

    for i, d in enumerate(pbar):
        
        input_values, token_ids, token_seq_lengths, lang_ids, lang_labels_lengths = d
        lang_ids = [ids[:length].tolist() for (length, ids) in zip(lang_label_lengths, lang_ids)]

        if input_values.shape[1]>config.max_audio_len:
            print("skipping batch : ", i)
            continue

        logits = model(input_values)
        token_logits, lang_logits = logits[...,:-3], logits[..., -3:]

        predicted_ids, predicted_langs = torch.argmax(token_logits, dim=-1).cpu(), torch.argmax(lang_logits, dim=-1).cpu()
        transcriptions, predicted_langs = tokenizer.batch_decode(predicted_ids), tokenizer.batch_decode(predicted_langs)
        predicted_langs = [[1 if elem=='s' or elem=='<s' else 2 for elem in predicted.split('><')] for predicted in predicted_langs]

        if config.transliterate:
            transcriptions = tokenizer.revert_transliteration(transcriptions, predicted_langs)
    
        references = tokenizer.batch_decode(token_ids)
        
        if i==show_sample_no or i==0:
            print("Sample prediction: ", transcriptions)
            print("Sample reference: ", references)
        
        wer_metric.add_batch(predictions=transcriptions, 
                             references=references)
        
        acc_metric.add_batch(predictions=predicted_langs,
                             references=lang_ids)
    
    wer_score = wer_metric.compute()
    acc_score = acc_metric.comput()
    print("WER Evaluation metric: ", wer_score)
    print("Language prediction accuracy: ", acc_score)
    return config.asr_param*wer_score + config.lang_param*acc_score


def collate_fn(batch, tokenizer):
    speech_lis = [elem["speech"] for elem in batch]
    text_lis = [elem["text"].upper() for elem in batch]

    input_values = tokenizer(speech_lis, return_tensors="pt", 
                             padding='longest').input_values

    token_ids, token_seq_lengths, lang_ids, lang_labels_lengths = tokenizer.batch_tokenize(text_lis)
    
    return (input_values.to(config.device), token_ids.to(config.device), token_seq_lengths.to(config.device),
            lang_ids.to(config.device), lang_labels_lengths.to(config.device))

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

    train_dataset, val_dataset, train_dataset, mono_dataset = get_datasets()

    if(config.train):
        mono_dataloader = torch.utils.data.DataLoader(dataset=mono_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params) if mono_dataset is not None else None
        
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)

        train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, mono_dataloader)
    
    if(config.eval):
        print(compute_metric(model, tokenizer, test_dataloader))
    
    print("TRAINING DONE!")
