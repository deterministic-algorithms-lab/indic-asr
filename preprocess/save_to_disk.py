import argparse
from datasets import load_dataset
import os

all_datasets = ['/home/krishnarajule3/ASR/data/Bengali-English',
                '/home/krishnarajule3/ASR/data/Gujarati',
                '/home/krishnarajule3/ASR/data/Hindi'
                '/home/krishnarajule3/ASR/data/Hindi-English',
                '/home/krishnarajule3/ASR/data/Marathi',
                '/home/krishnarajule3/ASR/data/Odia',
                '/home/krishnarajule3/ASR/data/Tamil',
                '/home/krishnarajule3/ASR/Telugu']

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', default=all_datasets, type=str, nargs='+')

args = parser.parse_args()
for dir in args.dirs:
    ds = load_dataset('/home/datasets/code_switch_asr', data_dir=dir, writer_batch_size=1000)
    ds.save_to_disk(os.path.join('bucket', args.split('/')[-1])

