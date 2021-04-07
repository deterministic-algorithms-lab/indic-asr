from typing import List,Tuple,Union
import re
import unicodedata

import torch
import torch.nn as nn
from transformers import Wav2Vec2Tokenizer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import enchant
from nltk.corpus import indian

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
        self.hindi_words = [unicodedata.normalize('NFKC', word) for word in nltk.corpus.indian.words('hindi.pos')]
        
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
    
    def back_transliterate_word(self, word: str, predicted_lang_id=None):
        word = word.lower()
        if not self.en_dict.check(word) and predicted_lang_id!=1:
            word = unicodedata.normalize('NFKC', word)
            word = transliterate(word, sanscript.KOLKATA, sanscript.DEVANAGARI)
        else:
            transliterated_word = unicodedata.normalize('NFKC', word)
            transliterated_word = transliterate(transliterated_word, sanscript.KOLKATA, sanscript.DEVANAGARI)
            if unicodedata.normalize('NFKC', transliterated_word) in self.hindi_words:
                word = transliterated_word
        return unicodedata.normalize('NFKC', word).upper()
        
    def revert_transliteration(self, texts: List[str], lang_ids: List[List[int]])->str:
        
        back_transliterated_texts = []

        for (text, langs) in zip(texts, lang_ids):
            words = text.split()
            if len(langs)<len(words):
                langs = langs + [None]*(len(words)-len(langs))
            back_transliterated = [self.back_transliterate_word(word, lang) for word, lang in zip(words, langs)]
            back_transliterated_texts.append(' '.join(back_transliterated))
        
        return back_transliterated_texts
    
    def get_lang_ids(self, text: str) -> List[int]:
        return [1 if word.encode().isalpha() else 2 for word in text.split()]
    
    def tokenize(self, text: str, **kwargs) -> Union[Tuple[List[int],List[int]],List[int]]:
        """
        Converts a single str into a sequence of token ids.
        """
        text=text.upper()
        text = ' '.join(text.split())
        
        lang_ids = self.get_lang_ids(text)
        
        if config.transliterate:
            text = self.transliterate(text)           
        
        text = text.replace(' ', self.word_delimiter_token)
        tokens = [self.bos_token_id]
        
        for char in text:
            tokens.append(self._convert_token_to_id_with_added_voc(char))

        tokens.append(self.eos_token_id)
        
        return tokens, lang_ids
    
    def pad_batch_sentences(self, sentences_word: Union[List[List[int]],List[Tuple[List[int],List[int]]]], max_length: int=-1) -> Union[Tuple[torch.FloatTensor, torch.IntTensor],Tuple[torch.FloatTensor, torch.IntTensor,torch.FloatTensor, torch.IntTensor]]:
        """
        Pads all list of token ids, in a batch to the maximum length.
        Truncates all sequences to max_length.
        """
        
        sentences = [sentence[:max_length] for sentence in sentences]
        lengths = [len(sentence) for sentence in sentences]
        max_len = max(lengths)
        
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence + [self.pad_token_id]*(max_len-len(sentence))
        
        return torch.tensor(sentences, dtype=torch.float32), torch.tensor(lengths)
      
    def batch_tokenize(self, texts: List[str], **kwargs) -> Union[Tuple[torch.FloatTensor, torch.IntTensor],Tuple[torch.FloatTensor, torch.IntTensor,torch.FloatTensor, torch.IntTensor]]:
        """
        Tokenizes and batches together a list of texts
        """
        tokenizer_output = [self.tokenize(sentence) for sentence in text]
        padded_token_ids, token_seq_lengths = self.pad_batch_sentences([elem[0] for elem in tokenizer_output])
        padded_lang_ids, lang_labels_lengths = self.pad_batch_sentences([elem[1] for elem in tokenizer_output])
        return padded_token_ids, token_seq_lengths, padded_lang_ids, lang_labels_lengths