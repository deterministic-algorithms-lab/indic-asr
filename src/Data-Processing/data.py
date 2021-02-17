# Data Processing
#
from pydub import AudioSegment
from tqdm import tqdm
from __future__ import absolute_import, division, print_function
import datasets

# Data Pre-processing, splitting the Chapters into audio chunks/segments
#
class pre_processing:

    def __init__(self, segments_file, audio_path, audio_type, new_audio_path):

        self.segments_file = segments_file             # <segments> file along with the path
        self.chap_audio_path = chap_audio_path         # Path to chapter audio files
        self.audio_type = audio_type                   # The type of the audio file initially and the type of the audio segments
        self.new_audio_path = new_audio_path           # Path to the generated audio files

    def pre_processing(self):
        # Using "segments" file to generate audio chunks
        # Loop through the "segments" file to get the timestamps, audio ids and the segment names
        fh = open(self.segments_file, "r").readlines()
        for i in tqdm(fh):
            l = list([])
            l = i.split(" ")
            seg = l[0]                                  # Segment Id
            aid = l[1]                                  # Audio Id
            s_ts = float(l[2])*1000                            # Start Time(ms)
            e_ts = float(l[3])*1000                            # End Time(ms)

            # Audio segment generator
            a_seg = AudioSegment.from_wav(self.audio_path + "/" + aid + "." + self.audio_type)
            a_seg = a_seg[s_ts:e_ts]
            a_seg.export(self.new_audio_path + "/" + seg + "." + self.audio_type, format=self.audio_type)


# Data Loader, using HuggingFace "datasets"
#
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
                    "audio_chunks": audio_path+audio_file,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "serial_id": serial_id,
                    "text": text,
                }
                yield key, example
