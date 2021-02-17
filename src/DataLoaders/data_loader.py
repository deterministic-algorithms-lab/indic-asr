# Data Loader using "datasets"
#
from __future__ import absolute_import, division, print_function
import os
import datasets

_CITATION = ""
_DESCRIPTION = "Interspeech 2021 Special Session - Multilingual and code-switching ASR challenges for low resource Indian languages"
_URL = ""

class ASRConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(ASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class ASR(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audio_file_path": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "chapter_id": datasets.Value("string"),
                    "serial_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=("audio_file_path", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        text_path_train = "/home/krishnarajule3/ASR/data/Hindi-English/train/transcript/text"
        text_path_dev = "/home/krishnarajule3/ASR/data/Hindi-English/dev/transcript/text"
        #text_path_test = "/home/krishnarajule3/ASR/data/Hindi-English/test/transcript/text"

        audio_path_train = "/home/krishnarajule3/ASR/data/Hindi-English/train/audio/"
        audio_path_dev = "/home/krishnarajule3/ASR/data/Hindi-English/dev/audio/"
        #audio_path_test = "/home/krishnarajule3/ASR/data/Hindi-English/test/audio/"

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"text_path": text_path_train, "audio_path": audio_path_train}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"text_path": text_path_dev, "audio_path": audio_path_dev}),
            #datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"archive_path": text_path_test, "audio_path": audio_path_test}),
        ]

    # This function returns the examples in the raw (text) form
    def _generate_examples(self, text_path, audio_path):
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                key, text = line.split(" ", 1)
                audio_file = "{key}.wav".format(key=key)
                speaker_id, chapter_id, serial_id = [i for i in key.split("_")]
                example = {
                    "audio_file_path": audio_path+audio_file,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "serial_id": serial_id,
                    "text": text,
                }
                yield key, example
