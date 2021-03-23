from typing import List,Tuple,Union
import re
import unicodedata

import torch
import torch.nn as nn
from transformers import Wav2Vec2Tokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import enchant

from configs import config
class Wav2Vec2Tok(Wav2Vec2Tokenizer):
    """
    Extending the base tokenizer of Wav2Vec2 for the purpose of encoding
    text sequences. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not config.transliterate:
            for i in range(2304, 2432) :
                self._add_tokens(chr(i))
        else:
            self.en_dict = enchant.Dict("en_US")
            for elem in ['̄', '̣', '̐', '́', '़', "'ॉ", '̃', '_', 'ऑ', '^', '…', '°', '̂', '̱',  'ॅ', 'ऍ', ':']:
                self._add_tokens(elem)
        self.mappings = {'$': ' dollar ', '@' : ' at the rate ', '+': ' plus ', '<':' less than ', '>' : ' greater than ', '&' : ' and ', '%':' percent '}

    def normalize(self, text):
        """
        Replaces common symbols like @, $ with their phonetic forms.                  
        Performs unicode NFD normalization, so that the base characters 
        and diacritics are separated in the output.
        """
        text = unicodedata.normalize('NFD',text)
        for k,v in self.mappings.items():
            text = text.replace(k, v)
        return ' '.join(text.split())

    def transliterate(self, text: str)-> str:
        transliteration = transliterate(text, sanscript.DEVANAGARI, sanscript.KOLKATA)
        return self.normalize(transliteration).upper()
    

    def remove_sos(self, texts: List[str]) -> List[str]:
        processed_texts = []
        for text in texts:
            processed_texts.append(text[3:] if text.startswith('<S>') else text)
        return processed_texts
    
    def revert_transliteration(self, texts: List[str])->str:
        if not config.transliterate:
            return self.remove_sos([text.upper() for text in texts])

        back_transliterated_texts = []
        for text in texts:
            text = text.lower()
            text = text.split()
            reverted_text = []
            for elem in text:
                if not self.en_dict.check(elem):
                    elem = unicodedata.normalize('NFKC', elem)
                    reverted_elem = transliterate(elem, sanscript.KOLKATA, sanscript.DEVANAGARI)
                    if re.search('[a-zA-Z]',reverted_elem) is not None:
                        reverted_elem = elem
                reverted_text.append(reverted_elem)
            reverted_text = ' '.join(reverted_text) 
            back_transliterated_texts.append(unicodedata.normalize('NFKC', reverted_text).upper())
        
        back_transliterated_texts = self.remove_sos(back_transliterated_texts)
        return back_transliterated_texts

    def tokenize(self, text: str, **kwargs) -> Union[Tuple[List[int],List[int]],List[int]]:
        """
        Converts a single str into a sequence of token ids.
        """
        text=text.upper()
        text = ' '.join(text.split())
        words_lang=[]
        if config.language_identification or config.language_identification_asr:
            for word in text.split():               
                if(word.encode().isalpha()):
                    words_lang.append(1)
                else:
                    words_lang.append(2)
          
        
        if config.transliterate:
            text = self.transliterate(text)           
        
        text = text.replace(' ', self.word_delimiter_token)
        tokens = [self.bos_token_id]
        
        for char in text:
            tokens.append(self._convert_token_to_id_with_added_voc(char))

        tokens.append(self.eos_token_id)
        
        if config.language_identification:
            return words_lang
        
        if config.language_identification_asr:
            return tokens,words_lang
        
        return tokens
    
    def pad_batch_sentences(self, sentences_word: Union[List[List[int]],List[Tuple[List[int],List[int]]]], max_length: int=None) -> Union[Tuple[torch.FloatTensor, torch.IntTensor],Tuple[torch.FloatTensor, torch.IntTensor,torch.FloatTensor, torch.IntTensor]]:
        """
        Pads all list of token ids, in a batch to the maximum length.
        Truncates all sequences to max_length.
        """
        
        if not config.language_identification_asr:
            sentences = [sentence[:max_length] for sentence in sentences_word]
            lengths = [len(sentence) for sentence in sentences_word]
            max_len = max(lengths)

            for i, sentence in enumerate(sentences):
                sentences[i] = sentence + [self.pad_token_id]*(max_len-len(sentence))        

            return torch.tensor(sentences, dtype=torch.float32), torch.tensor(lengths)
        
        sentences = [sentence[0][:max_length] for sentence in sentences_word]
        lengths = [len(sentence[0]) for sentence in sentences_word]
        max_len = max(lengths)

        words = [sentence[1][:max_length] for sentence in sentences_word]
        w_lengths = [len(sentence[1]) for sentence in sentences_word]
        w_max_len = max(w_lengths)

        for i, (sentence,word) in enumerate(zip(sentences,words)):
                sentences[i] = sentence + [self.pad_token_id]*(max_len-len(sentence))  
                words[i]=word+ [self.pad_token_id]*(w_max_len-len(word))
        
        return torch.tensor(sentences, dtype=torch.float32), torch.tensor(lengths),torch.tensor(words, dtype=torch.float32), torch.tensor(w_lengths)
    
    def batch_tokenize(self, texts: List[str], **kwargs) -> Union[Tuple[torch.FloatTensor, torch.IntTensor],Tuple[torch.FloatTensor, torch.IntTensor,torch.FloatTensor, torch.IntTensor]]:
        """
        Tokenizes and batches together a list of texts
        """
        tokenized_sentences = []
        for sentence in texts:
            tokenized_sentences.append(self.tokenize(sentence))
        return self.pad_batch_sentences(tokenized_sentences)
