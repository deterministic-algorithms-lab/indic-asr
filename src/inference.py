import torch
import numpy as np
import pandas as pd
import soundfile as sf

from datasets import load_dataset,load_metric

from configs import config
from model import get_model
from tokenizer import Wav2Vec2Tok
from train2 import compute_metric
from ctcdecode import CTCBeamDecoder

import argparse

def load_data(path):
    with open(path,'r',encoding='UTF-8') as file:
        data=file.read().split('\n')
    return data

def get_mappings(tokenizer):
    st=[]
    for i in range(len(tokenizer.get_vocab())):
        st.append(tokenizer.convert_ids_to_tokens(i))
    st[0]=''
    st[1]=''
    st[2]=''
    st[3]=''
    st[4]=' '
    return st
mappings={'$': ' dollar ', '@' : ' at the rate ', '+': ' plus ', '<':' less than ', '>' : ' greater than ', '&' : ' and ', '%':' percent '}
def make_preds(tokenizer,model,data,decoder,labels):
    ans=[]

    with torch.no_grad():
        for i in data:
            path=i.split('\t')[0]
            speech=sf.read(path)[0]
            input_values = tokenizer(speech, return_tensors="pt", 
                                        padding='longest').input_values.to(config.device)
            logits = model(input_values)
            logits=F.log_softmax(logits.logits,dim=-1)

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
            k=torch.argmax(beam_scores)
            text=beam_decode(labels,[beam_results[0][k]],[out_lens[0][k]])
            
            for k,v in mappings.items():
                text.replace(v.strip(),k)
            ans.append(text)

    return ans

def beam_decode(mapp,beams,length):
    ans=[]
    for b,l in zip(beams,length):
        a=''
        for j in range(l):
            a=a+mapp[b[j]]
        ans.append(a)
    return ans

def compute_metric(preds,data):
    metric=load_metric('wer')

    for pred,d in zip(preds,data):
        metric.add_batch(predictions=[pred], 
                         references=[d.split()[1]])
    
    score = metric.compute()
    print("Evaluation metric: ", score)
    return score

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Inference ASR, set the config accordingly!')
    parser.add_argument('--model_path',type=str,default=config.asr_model,
                        help='ASR model')

    parser.add_argument('--LM_path',type=str,default=config.lm_model,
                        help='LM model')

    parser.add_argument('--beam_size',type=int,default=2000,
                        help='Beam width')

    parser.add_argument('--alpha',type=float,default=0.75,
                        help='The weight α controls the relative contributions of the language model and the CTC network.')

    parser.add_argument('--Beta',type=float,default=0.25,
                        help='The weight β encourages more words in the transcription.')

    parser.add_argument('--file_path',type=str,default=config.file_path,help='file consisting of audios')

    parser.add_argument('--eval',action='store_true',help='Only if file consisting of audios and transcription')

    args = parser.parse_args()
    
    data=load_data(args.file_path)

    tokenizer=Wav2Vec2Tok.from_pretrained(config.model)
    model=get_model(tokenizer)
    model.load_state_dict(torch.load(args.model_path),strict=False)
    model=model.eval().to(config.device)

    labels=get_mappings(tokenizer)

    decoder = CTCBeamDecoder(
    labels=st,
    model_path=args.model_path,
    alpha=args.alpha,
    beta=args.beta,
    cutoff_top_n=400,
    cutoff_prob=1.0,
    beam_width=args.beam_size,
    num_processes=2,
    blank_id=0,
    log_probs_input=True)

    
    preds=make_preds(tokenizer,model,data,decoder,labels)

    if args.eval :
        score=compute_metric(preds,data)
    
    with open("./output.txt",'w',encoding='UTF-8') as file:
        for i,j in zip(preds,data):
            file.write(j+'\t'+i+' \n')
        if args.eval :
            file.write('WER : '+str(score))

    print("DONE!")