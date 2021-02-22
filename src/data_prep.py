# Data Pre-processing, splitting the Chapters into audio chunks/segments
#
from __future__ import absolute_import, division, print_function
from pydub import AudioSegment
from tqdm import tqdm

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
