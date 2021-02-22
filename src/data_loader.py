# Data Loader, using HuggingFace "datasets"
#
from __future__ import absolute_import, division, print_function
import datasets
from configs import config
from tqdm import tqdm

_CITATION = "--"
_DESCRIPTION = "Interspeech 2021 Special Session - Multilingual and code-switching ASR challenges for low resource Indian languages"
_URL = "--"

class ASRConfig(datasets.BuilderConfig):
    """BuilderConfig for ASR"""

    def __init__(self, **kwargs):
        super(ASRConfig, self).__init__(**kwargs)


class ASR(datasets.GeneratorBasedBuilder):
    """ASR: Interspeech 2021"""

    BUILDER_CONFIGS = [
        ASRConfig(name="asr_indian", description="Multilingual and code-switching ASR challenges for low resource Indian languages",),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audio_chunks": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "chapter_id": datasets.Value("string"),
                    "serial_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=("audio_chunks", "text"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"trans_path": config.train_trans, "audio_path": config.train_audio}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"trans_path": config.dev_trans, "audio_path": config.dev_audio}),
            #datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"trans_path": config.test_trans, "audio_path": config.test_audio}),
        ]

    def _generate_examples(self, trans_path, audio_path):
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                key, text = line.split(" ", 1)
                audio_file = "{key}.wav".format(key=key)
                speaker_id, chapter_id, serial_id = [i for i in key.split("_")]
                example = {
                    "audio_chunks": audio_path+audio_file,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "serial_id": serial_id,
                    "text": text,
                }
                yield key, example
