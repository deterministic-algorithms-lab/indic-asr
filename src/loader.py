# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""CodeSwitch automatic speech recognition dataset."""

from __future__ import absolute_import, division, print_function

import glob
import os

import datasets

import soundfile as sf

_CITATION = """\

"""

_DESCRIPTION = """\

Download dataset as:

```
!wget http://www.openslr.org/resources/104/Hindi-English_test.zip
!unzip -P e*F}[2]? Hindi-English_test.zip
!tar -xzf Hindi-English_test.tar.gz
```

Download and install datasets library as:
```
!git clone https://github.com/deterministic-algorithms-lab/datasets
!cd datasets
!git checkout code_switch-asr

!pip install .

!cd ..
```

Usage:

```python

from datasets import load_dataset
from datasets.utils.file_utils import DownloadConfig

ds = load_dataset(<folder containing this script>, 
                  data_dir= <location where data was downloaded>, 
                  download_config = DownloadConfig(local_files_only=True))
```

"""

_URL = "http://www.openslr.org/12"
_DL_URL = "http://www.openslr.org/resources/104/"

_DL_URLS = {
    "hi-en" : {"train": _DL_URL+'Hindi-English_train.zip',
               "test": _DL_URL+'Hindi-English_test.zip',},
    
    "bn-en": {"train": _DL_URL+'Bengali-English_train.zip',
              "test": _DL_URL+'Bengali-English_test.zip',},
}



_LANG_CODES = {
    "Tamil": "ta",
    "Gujarati": "gu",
    "Telegu" : "te",   
}

class CodeSwitchASRConfig(datasets.BuilderConfig):
    """BuilderConfig for CodeSwitchASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(CodeSwitchASRConfig, self).__init__(version=datasets.Version("0.0.0", ""), **kwargs)


class CodeSwitchASR(datasets.GeneratorBasedBuilder):
    """CodeSwitch dataset."""

    BUILDER_CONFIGS = [
        CodeSwitchASRConfig(name="hi-en", description="Hindi-English Code-switched speech."),
        CodeSwitchASRConfig(name="bn-en", description="Bengali-English Code-switched speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "speech": datasets.Sequence(feature=datasets.Value('float32')),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("speech", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators_simple(self, dl_manager):
        if self.config.name not in _DL_URLS:
            root = self.config.data_dir
            archive_path = {'train' : os.path.join(root, 'train/'),
                            'test': os.path.join(root, 'test/'),}
        else:
            archive_path = dl_manager.download_and_extract(_DL_URLS[self.config.name])
        
        train_splits = [
                datasets.SplitGenerator(name='train', gen_kwargs={"archive_path": archive_path['train']}),
            ]

        return train_splits + [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"archive_path": archive_path['test']}),
        ]
    
    def _split_generators_with_measurement(self, dl_manager):
        if self.config.name not in _DL_URLS:
            root = self.config.data_dir
            archive_path = {'train' : os.path.join(root, _LANG_CODES[root.split('/')[-1]]+'-in-Train/'),
                            'test': os.path.join(root, _LANG_CODES[root.split('/')[-1]]+'-in-Test/'),
                            'Measurement' : os.path.join(root, _LANG_CODES[root.split('/')[-1]]+'-in-Measurement/'),}
        else:
            archive_path = dl_manager.download_and_extract(_DL_URLS[self.config.name])

        train_splits = [
                datasets.SplitGenerator(name='train', gen_kwargs={"archive_path": archive_path['train']}),
            ]
        
        #meas_splits =  [
        #        datasets.SplitGenerator(name='Measurement', gen_kwargs={"archive_path": archive_path['Measurement']}),
        #    ]
        
        return train_splits + [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"archive_path": archive_path['test']}),
        ] #+ meas_splits
    
    def _split_generators(self, dl_manager):
        if '-' in self.config.name or self.config.data_dir.split('/')[-1] not in _LANG_CODES:
            return self._split_generators_simple(dl_manager)
        else:
            return self._split_generators_with_measurement(dl_manager)
        
    def make_idx_dicts(self, speaker_file, segments_file, text_file):
        self.speaker_to_idx = dict()
        with open(speaker_file) as f:
            for i, elem in enumerate(f.readlines()):
                self.speaker_to_idx[elem.strip()]=i
        
        self.chapter_to_idx = dict()
        with open(segments_file) as f:
            i=0
            for elem in f.readlines():
                chapter_id = elem.strip().split()[0].split('_')[1]
                if chapter_id not in self.chapter_to_idx:
                    self.chapter_to_idx[chapter_id]=i
                    i+=1
        
        self.id_to_text = dict()
        with open(text_file) as f:
            for elem in f.readlines():
                id, text = elem.strip().split(' ', 1)
                self.id_to_text[id] = text
        
    def _generate_examples_code_switch(self, archive_path):
        """Generate examples from a CodeSwitch archive_path."""
        segments_file = os.path.join(archive_path, 'transcripts/segments')
        speaker_file = os.path.join(archive_path, 'transcripts/spkr_list')
        text_file = os.path.join(archive_path, 'transcripts/text')

        self.make_idx_dicts(speaker_file, segments_file, text_file)

        with open(segments_file) as f:
            cur_file = None
            for line in f.readlines():
                line = line.strip().split()
                
                if cur_file != line[1]+'.wav':
                    speech, sr = sf.read(os.path.join(archive_path, line[1]+'.wav'))
                
                start_time, end_time = float(line[2]), float(line[3])
                
                example = {
                    "id" : line[0],
                    "speaker_id" : self.speaker_to_idx[line[0].split('_')[0]],
                    "chapter_id" : self.chapter_to_idx[line[1]],
                    "text" : self.id_to_text[line[0]],
                    "speech" : speech[int(start_time*sr):int(end_time*sr)]
                }
                
                yield line[0], example
    
    def _generate_examples_mono(self, archive_path):
        transcripts_file = os.path.join(archive_path, 'transcription.txt')
        with open(transcripts_file) as f:
            files_n_text = [line.strip().split(None,1) for line in f.readlines()]
            
            try:
                audio_dir = 'Audios'
                speech, sr = sf.read(os.path.join(archive_path, audio_dir, wav_file+'.wav'))
            except:
                audio_dir = 'audio'
            
            for wav_file, text in files_n_text:
                speech, sr = sf.read(os.path.join(archive_path, audio_dir, wav_file+'.wav'))
                example = {
                    "id" : wav_file,
                    "speaker_id" : -1,
                    "chapter_id" : -1,
                    "text" : text,
                    "speech" : speech,
                }
                
                yield wav_file, example
    
    def _generate_examples(self, archive_path):
        if '-' in archive_path.split('/')[-3]:
            for sample in self._generate_examples_code_switch(archive_path):
                yield sample
        else:
            for sample in self._generate_examples_mono(archive_path):
                yield sample