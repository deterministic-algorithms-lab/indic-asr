import os

class MonoData(Dataset):
    def __init__(self,path):
        self.path=path
        self.file=open(path+'/transcription.txt','r',encoding='UTF-8').read().split("\n")
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self,index):
        audio,text=self.file[index].split(' ',1)
        audio=self.path+'/Audios/'+audio
        return audio,text
