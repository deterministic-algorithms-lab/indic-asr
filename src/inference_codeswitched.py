import torch
import numpy as np
import pandas as pd
import soundfile as sf
import torch.nn.functional as F

from datasets import load_dataset,load_metric

from configs import config
from model import get_model
from tokenizer import Wav2Vec2Tok
from train2 import compute_metric
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm

import argparse

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
        for d in tqdm(val_data):

            sp,sr=sf.read(d["speech"])
            input_values = tokenizer(sp[sr*d['start']:sr*d['end']], return_tensors="pt", 
                                     padding='longest').input_values.to(config.device)
            
            
            logits = model(input_values)

            if config.language_identification_asr:
                logits1,logits2=F.log_softmax(logits[0].logits,dim=-1),F.log_softmax(logits[1].logits,dim=-1)
            else:
                logits1=F.log_softmax(logits.logits,dim=-1)

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits1)
            k=torch.argmax(beam_scores)
            transcriptions=beam_decode(labels,[beam_results[0][k]],[out_lens[0][k]])

            if config.language_identification_asr:
                words_id= torch.argmax(logits2, dim=-1).cpu()
                
                words_id= tokenizer.batch_decode(words_id)

                transcriptions = tokenizer.revert_transliteration(zip(transcriptions,words_id))
            else:
                if config.transliterate:
                    transcriptions = tokenizer.revert_transliteration(transcriptions)
            

            for k,v in mappings.items():
                transcriptions[0].replace(v.strip(),k)
            ans.append(transcriptions[0].lower())

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
    score=[]
    for pred,d in tqdm(zip(preds,data)):
        metric.add_batch(predictions=[pred], 
                         references=[d['text']])
        score.append(metric.compute())
    score = sum(score)/len(score)
    print("Evaluation metric: ", score)
    return score

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Inference ASR, set the config accordingly!')
    parser.add_argument('--model_path',type=str,default=config.asr_model,
                        help='ASR model')

    parser.add_argument('--lm_path',type=str,default=config.lm_model,
                        help='LM model')

    parser.add_argument('--beam_size',type=int,default=1,
                        help='Beam width')

    parser.add_argument('--alpha',type=float,default=0.75,
                        help='The weight α controls the relative contributions of the language model and the CTC network.')

    parser.add_argument('--beta',type=float,default=0.25,
                        help='The weight β encourages more words in the transcription.')

    parser.add_argument('--eval',action='store_true',help='Only if file consisting of audios and transcription')

    args = parser.parse_args()
    
    tokenizer=Wav2Vec2Tok.from_pretrained(config.model)
    model=get_model(tokenizer)
    model.load_state_dict(torch.load(args.model_path),strict=False)
    model=model.eval().to(config.device)

    labels=get_mappings(tokenizer)

    decoder = CTCBeamDecoder(
    labels=labels,
    model_path=args.lm_path,
    alpha=args.alpha,
    beta=args.beta,
    cutoff_top_n=400,
    cutoff_prob=1.0,
    beam_width=args.beam_size,
    num_processes=2,
    blank_id=0,
    log_probs_input=True)

    val_data=load_dataset(config.data_loading_script,data_dir=config.data_dir,split='test').filter(lambda x:x['end']-x['start']>0)

    preds=make_preds(tokenizer,model,val_data,decoder,labels)

    with open("./output.txt",'w',encoding='UTF-8') as file:
        for i,j in tqdm(zip(preds,val_data)):
            file.write(j['speech']+'\t'+j['text']+'\t'+i+' \n')
    
    if args.eval :    
        with open("./metric.txt",'w',encoding='UTF-8') as file:
            
            score=compute_metric(preds,val_data)
            
            if args.eval :
                file.write('WER : '+str(score))

    print("DONE!")
