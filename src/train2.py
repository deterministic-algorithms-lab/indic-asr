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
    model.load_state_dict(torch.load(config.prev_checkpoint+"/pytorch_model.bin"),strict=False)
    print("model loaded!")
    return model


def train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataset, mono_dataloader=None):
    
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

        loss=0
        epoch_loss = 0
        
        if mono_dataloader is not None:
            pbar=tqdm(itertools.chain(train_dataloader, mono_dataloader), desc="Training epoch %d"%(epoch))
        else:
            pbar=tqdm(train_dataloader, desc="Training epoch %d"%(epoch))
        
        for i, d in enumerate(pbar):
            pbar.set_postfix(loss =loss)
            
            iters+=1
            input_values, labels1, label_lengths1,labels2, label_lengths2=None,None,None,None,None
            if config.language_identification_asr:
                input_values, labels1, label_lengths1,labels2, label_lengths2 = d
            
            else:
                input_values, labels1, label_lengths1=d
            # print(labels1,labels2)
            if input_values.shape[1]>config.max_audio_len:
                print("skipping batch : ", i)
                continue
            
            optimizer.zero_grad()
            
            logits1,logits2=None,None
            logits = model(input_values)
            
            if config.language_identification_asr:
                logits1,logits2=F.log_softmax(logits[0].logits,dim=-1),F.log_softmax(logits[1].logits,dim=-1)
            else:
                logits1=F.log_softmax(logits.logits,dim=-1)
            loss = ctc_loss(logits1.transpose(0,1), labels1, 
                            find_lengths(logits1, tokenizer.pad_token_id), label_lengths1)
            
            if config.language_identification_asr:
                loss=loss+config.lang_param*ctc_loss(logits2.transpose(0,1), labels2, 
                            find_lengths(logits2, tokenizer.pad_token_id), label_lengths2)
            # print("Training loss : ", loss)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            loss = loss.item()
            epoch_loss += loss
            
            if(iters%config.num_iters_checkpoint==0):
                model.eval()
                
                val_losses=eval_model(model, tokenizer, val_dataloader)
                
                wer_score.append(compute_metric(model, tokenizer, test_dataset))
                
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
        
        input_values, labels1, label_lengths1,labels2, label_lengths2=None,None,None,None,None
        if config.language_identification_asr:
            input_values, labels1, label_lengths1,labels2, label_lengths2 = d
        else:
            input_values, labels1, label_lengths1=d

        logits1,logits2=None,None
        logits = model(input_values)

        if config.language_identification_asr:
            logits1,logits2=F.log_softmax(logits[0].logits,dim=-1),F.log_softmax(logits[1].logits,dim=-1)
        else:
            logits1=F.log_softmax(logits.logits,dim=-1)
        
        loss = ctc_loss(logits1.transpose(0,1), labels1, 
                            find_lengths(logits1, tokenizer.pad_token_id), label_lengths1)
            
        if config.language_identification_asr:
            loss=loss+config.lang_param*ctc_loss(logits2.transpose(0,1), labels2, 
                        find_lengths(logits2, tokenizer.pad_token_id), label_lengths2)
        
        loss = loss.item()
        
        epoch_loss += loss

    print("Mean validation loss:", (epoch_loss / num_valid_batches))
    return (epoch_loss / num_valid_batches)

def compute_metric(model, tokenizer, test_dataset):
    
    metric = load_metric('wer')

    pbar = tqdm(test_dataset, desc="Computing metric")

    show_sample_no = random.randint(1, len(test_dataset)-1)

    for i, d in enumerate(pbar):
        
        sp,sr=sf.read(d["speech"])
        input_values = tokenizer(sp[sr*d['start']:sr*d['end']], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)
        
        logits1,logits2=None,None
        logits = model(input_values)

        if config.language_identification_asr:
            logits1,logits2=F.log_softmax(logits[0].logits,dim=-1),F.log_softmax(logits[1].logits,dim=-1)
        else:
            logits1=F.log_softmax(logits.logits,dim=-1)


        predicted_ids = torch.argmax(logits1, dim=-1).cpu()
        transcriptions = tokenizer.batch_decode(predicted_ids)


        if config.language_identification:
            
            print("Sample prediction: ", transcriptions[0].replace('<s>','1').replace('</s>','2'))
            print("Sample reference: ", d['text'].upper())
            return 
       

        if config.language_identification_asr:
            words_id= torch.argmax(logits2, dim=-1).cpu()
            
            words_id= tokenizer.batch_decode(words_id)
            transcriptions = tokenizer.revert_transliteration(zip(transcriptions,words_id))
        else:
            if config.transliterate:
               transcriptions = tokenizer.revert_transliteration(transcriptions)
        
        reference = d['text'].upper()
        
        if i==show_sample_no or i==0:
            print("Sample prediction: ", transcriptions)
            print("Sample reference: ", reference)
        
        metric.add_batch(predictions=transcriptions, 
                         references=[reference])
    
    score = metric.compute()
    print("Evaluation metric: ", score)
    return score

# def compute_metric(model, tokenizer, test_dataset):
#     metric = load_metric('wer')

#     pbar = tqdm(test_dataset, desc="Computing metric")

#     # show_sample_no = random.randint(1, len(test_dataset)-1)
#     show_sample_no=0
#     data=[]
#     for i, d in enumerate(pbar):
#         sp,sr=sf.read(d["speech"])
#         input_values = tokenizer(sp[sr*d['start']:sr*d['end']], return_tensors="pt", 
#                                      padding='longest').input_values.to(config.device)

        
#         logits = torch.nn.functional.log_softmax(model(input_values).logits,dim=-1)
#         # logits=model(input_values).logits

#         predicted_ids = torch.argmax(logits, dim=-1).cpu()
#         print(predicted_ids)
#         transcriptions = tokenizer.batch_decode(predicted_ids)
#         # transcriptions = tokenizer.revert_transliteration(transcriptions)
        
#         reference = d['text']
        
#         if i==show_sample_no or i==0:
#             print("Sample prediction: ", transcriptions[0])
#             print("Sample reference: ", reference)
        
#         data.append((transcriptions[0],reference))
#     return data
    #     metric.add_batch(predictions=transcriptions, 
    #                      references=[reference])
    
    # score = metric.compute()
    # print("Evaluation metric: ", score)
    # return score



# def collate_fn(batch, tokenizer):
#     speech_lis = [elem["speech"] for elem in batch]
#     text_lis = [elem["text"].upper() for elem in batch]
    
#     input_values = tokenizer(speech_lis, return_tensors="pt", 
#                                      padding='longest').input_values

#     if config.language_identification_asr:
#         labels1, label_lengths1,labels2, label_lengths2 = tokenizer.batch_tokenize(text_lis)
#         return (input_values.to(config.device), labels1.to(config.device), label_lengths1.to(config.device),
#                 labels2.to(config.device), label_lengths2.to(config.device))
    
#     labels, label_lengths = tokenizer.batch_tokenize(text_lis)

#     return (input_values.to(config.device), labels.to(config.device), label_lengths.to(config.device))

def collate_fn(batch, tokenizer):
    speech_lis=[]
    for elem in batch:
        sp,sr=sf.read(elem["speech"])
        speech_lis.append(sp[sr*elem['start']:sr*elem['end']])
    text_lis=[elem['text'] for elem in batch]
    input_values = tokenizer(speech_lis, return_tensors="pt", 
                                     padding='longest').input_values

    if config.language_identification_asr:
        labels1, label_lengths1,labels2, label_lengths2 = tokenizer.batch_tokenize(text_lis)
        return (input_values.to(config.device), labels1.to(config.device), label_lengths1.to(config.device),
                labels2.to(config.device), label_lengths2.to(config.device))
    
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

    train_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[2%:]", writer_batch_size=1000).filter(lambda x:x['end']-x['start']>0)
    val_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="train[:2%]", writer_batch_size=1000).filter(lambda x:x['end']-x['start']>0)
    test_dataset = load_dataset(config.data_loading_script, data_dir=config.data_dir, split="test", writer_batch_size=1000).filter(lambda x:x['end']-x['start']>0)
    
    if config.use_monolingual:
        mono_dataset = load_dataset(config.data_loading_script, data_dir=config.monolingual_data_dir, split="train", writer_batch_size=1000).filter(lambda x:x['end']-x['start']>0)
        mono_dataloader = torch.utils.data.DataLoader(dataset=mono_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
    else:
        mono_dataloader = None

    if(config.train):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, collate_fn= lambda b: collate_fn(b, tokenizer), **params)
        train_model(model, tokenizer, train_dataloader, val_dataloader, test_dataset, mono_dataloader)
    
    if(config.eval):
        print(compute_metric(model, tokenizer, test_dataset))
    
    print("TRAINING DONE!")
