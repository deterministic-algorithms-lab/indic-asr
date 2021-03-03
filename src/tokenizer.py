import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import Wav2Vec2Tokenizer
from configs import config
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
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
            for elem in ['̄', '̣', '̐', '́', '़', "'ॉ", '̃', '_', 'ऑ', '^', '…', '°', '̂', '̱',  'ॅ', 'ऍ', ':']:
                self._add_tokens(elem)
    
    def normalize(self, text):
        """
        Replaces common symbols like @, $ with their phonetic forms.                  
        Performs unicode NFD normalization, so that the base characters 
        and diacritics are separated in the output.
        """
        text = unicodedata.normalize('NFD',text)
        self.mappings = {'$': ' dollar ', '@' : ' at the rate ', '+': ' plus ', '<':' less than ', '>' : ' greater than ', '&' : ' and ', '%':' percent '}
        for k,v in self.mappings.items():
            text = text.replace(k, v)
        return ' '.join(text.split())

    def transliterate(self, text: str)-> str:
        transliteration = transliterate(data, sanscript.DEVANAGARI, sanscript.KOLKATA)
        return self.normalize(transliteration)clean_text()

    def tokenize(self, text: str, **kwargs) -> List[int]:
        """
        Converts a single str into a sequence of token ids.
        """
        if config.transliterate:
            text = self.transliterate(text)

        text = ' '.join(text.split())
        text = text.replace(' ', self.word_delimiter_token)
        tokens = [self.bos_token_id]
        
        for char in text:
            tokens.append(self._convert_token_to_id(char))

        tokens.append(self.eos_token_id)
        return tokens
    
    def pad_batch_sentences(self, sentences: List[List[int]], max_length: int=-1) -> Tuple[torch.FloatTensor, torch.IntTensor]:
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
    
    def batch_tokenize(self, texts: List[str], **kwargs) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Tokenizes and batches together a list of texts
        """
        tokenized_sentences = []
        for sentence in texts:
            tokenized_sentences.append(self.tokenize(sentence))
        return self.pad_batch_sentences(tokenized_sentences)
