# Code to split the Chapters into audio segments using in the transcript/text
#
from pydub import AudioSegment
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sf', '--segments_file', type=str,
                    required=True,help='<segments> file along with the path')
parser.add_argument('-ap', '--audio_path', type=str,
                    required=True,help='Path to the audio files')
parser.add_argument('-at', '--audio_type', type=str,
                    required=True,help='The type of the audio file initially and the type of the audio segments')
parser.add_argument('-nap', '--new_audio_path', type=str,
                    required=True,help='Path to the generated audio files')
args = parser.parse_args()

# Using "segments" file to generate audio chunks
# Loop through the "segments" file to get the timestamps, audio ids and the segment names
fh = open(args.segments_file, "r").readlines()
for i in tqdm(fh):
    l = list([])
    l = i.split(" ")
    seg = l[0]                                  # Segment Id
    aid = l[1]                                  # Audio Id
    s_ts = float(l[2])*1000                            # Start Time(ms)
    e_ts = float(l[3])*1000                            # End Time(ms)

    # Audio segment generator
    a_seg = AudioSegment.from_wav(args.audio_path + "/" + aid + "." + args.audio_type)
    a_seg = a_seg[s_ts:e_ts]
    a_seg.export(args.new_audio_path + "/" + seg + "." + args.audio_type, format=args.audio_type)
