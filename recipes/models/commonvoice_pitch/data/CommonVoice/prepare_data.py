''' COMMON VOICE DATA PREPARATION FOR WAV2LETTER 
This script takes the path to Common Voice dataset and returns
the desired partition lists (train.lst, dev.lst and test.lst) as specified in the arguments.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import sox
import ntpath
import string
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def check_speaker_separation(dataframe_a, dataframe_b):
    found_same_speakers = False

    dict_dataframe_a = get_class_distribution(dataframe_a, 'client_id')
    dict_dataframe_b = get_class_distribution(dataframe_b, 'client_id')

    for key, value in dict_dataframe_a.items():
        if key in dict_dataframe_b:
            found_same_speakers = True
            print("Error! Client ID " + key + " found in two different partitions.")

    for key, value in dict_dataframe_b.items():
        if key in dict_dataframe_a:
            found_same_speakers = True
            print("Error! Client ID " + key + " found in two different partitions.")

    if not found_same_speakers:
        print("Speakers are separated from both datasets!")

def findtranscriptfiles(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".wrd"):
                files.append(os.path.join(dirpath, filename))
    return files

def find_stratified_partition(dataframe, stratification_threshold, patience=50):
    seed = 0
    for i in range(patience):
        train_df, dev_df = split_dataset(dataframe, 'client_id', test_size=0.2, random_state=seed)
        dev_df, test_df = split_dataset(dev_df, 'client_id', test_size=0.39, random_state=seed)

        if not check_stratification(dataframe, train_df, 'age', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, train_df, 'gender', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, train_df, 'accent', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, dev_df, 'age', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, dev_df, 'gender', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, dev_df, 'accent', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, test_df, 'age', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, test_df, 'gender', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        if not check_stratification(dataframe, test_df, 'accent', stratification_threshold=stratification_threshold):
            seed += 1
            continue
        break

    print('Check partition sizes:')
    total_size = len(train_df) + len(dev_df) + len(test_df)
    print('Total size:')
    print(str(total_size))
    print('Train set size:')
    print(len(train_df)/total_size)
    print('Dev set size:')
    print(len(dev_df)/total_size)
    print('Test set size:')
    print(len(test_df)/total_size)
    print('Seed')
    print(seed)

    return train_df, dev_df, test_df, seed

    #dev_multi_occurrence_df, dev_single_occurrence_df = split_df_by_occurrences(dev_df)
    #dev_df, test_df = split_dataset(dev_multi_occurrence_df, 'client_id', test_size=0.5, random_state=0)
    #dev_df, test_df = train_test_split(dev_multi_occurrence_df, test_size=0.5, stratify=dev_multi_occurrence_df['client_id'].dropna().values, random_state=0)

def extract_sampleid(filepath):
    return ntpath.basename(filepath).split('.')[0]

def get_class_distribution(dataframe, class_name, normalize=True):
    return dataframe[class_name].value_counts(normalize=normalize, dropna=False).to_dict()

def split_df_by_occurrences(dataframe):
    recordings_by_speaker = get_class_distribution(dataframe, 'client_id', normalize=False)

    multi_occurrence_df = dataframe.copy()
    single_occurrence_df = dataframe.copy()

    for key, value in recordings_by_speaker.items():
        if int(value) == 1:
            multi_occurrence_df.drop(multi_occurrence_df.loc[multi_occurrence_df['client_id'] == key].index, inplace=True)
        else:
            single_occurrence_df.drop(single_occurrence_df.loc[single_occurrence_df['client_id'] == key].index, inplace=True)
    
    return multi_occurrence_df, single_occurrence_df

def split_dataset(dataframe, group_name, test_size=0.2, random_state=0):
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state).split(dataframe, groups=dataframe[group_name]))

    train_df = dataframe.iloc[train_inds]
    test_df = dataframe.iloc[test_inds]

    return train_df, test_df

def check_stratification(dataframe_a, dataframe_b, class_name, stratification_threshold=10):
    well_stratified = True
    dict_dataframe_a = get_class_distribution(dataframe_a, class_name, normalize=True)
    dict_dataframe_b = get_class_distribution(dataframe_b, class_name, normalize=True)

    for key, value in dict_dataframe_a.items():
        if key in dict_dataframe_b:
            difference = abs(value - dict_dataframe_b[key])*100
            if difference > stratification_threshold:
                well_stratified = False
            #print(key)
            #print(difference)

    #print(dict_dataframe_a)
    #print(dict_dataframe_b)

    return well_stratified

def get_default_partition(args):
    '''Get wav2letter partition lists for Common Voice proposed partitions, from train.tsv, dev.tsv and test.tsv.'''
    train_tsv_path = args.src + 'train.tsv'
    dev_tsv_path = args.src + 'dev.tsv'
    test_tsv_path = args.src + 'test.tsv'
    validated_tsv_path = args.src + 'validated.tsv'

    train_tsv = open(train_tsv_path, 'r', encoding='utf-8')
    dev_tsv = open(dev_tsv_path, 'r', encoding='utf-8')
    test_tsv = open(test_tsv_path, 'r', encoding='utf-8')
    validated_tsv = open(validated_tsv_path, 'r', encoding='utf-8')

    tsv_to_lst(args, train_tsv, args.dst + 'train.lst', data_partition='train')
    tsv_to_lst(args, dev_tsv, args.dst + 'dev.lst', data_partition='dev')
    tsv_to_lst(args, test_tsv, args.dst + 'test.lst', data_partition='test')
    tsv_to_lst(args, validated_tsv, args.dst + 'validated.lst', data_partition='valid')

def get_equivalidated_partition(args):
    '''Get wav2letter partition lists for a partition extracted from validated.tsv, with equal distribution in gender, accent and age.
    Step 1) Remove recordings with more down votes than the maximum set in the arguments (max_downvotes).
    Step 2) Split speakers separately, 80-10-10% (train-dev-test).
    Step 2a) Ensure equal distribution in age.
    Step 2b) Ensure equal distribution in gender.
    Step 2c) Ensure equal distribution in accent.
    '''
    validated_tsv_path = args.src + 'validated.tsv'
    validated_df = pd.read_csv(validated_tsv_path, sep='\t', header=0)

    print('Removing recordings with more than ' + str(args.max_downvotes) + ' down votes...')
    validated_df.drop(validated_df[validated_df.down_votes > args.max_downvotes].index, inplace=True)

    print('Iterate until finding a well stratified partition:')
    train_df, dev_df, test_df, seed = find_stratified_partition(validated_df, args.stratification_threshold, args.stratification_patience)

    train_df.to_csv('./train_stratified.tsv', sep='\t', index=None)
    dev_df.to_csv('./dev_stratified.tsv', sep='\t', index=None)
    test_df.to_csv('./test_stratified.tsv', sep='\t', index=None)

    check_speaker_separation(train_df, dev_df)
    check_speaker_separation(dev_df, test_df)

    train_tsv_path = args.src + 'train_stratified.tsv'
    dev_tsv_path = args.src + 'dev_stratified.tsv'
    test_tsv_path = args.src + 'test_stratified.tsv'

    train_tsv = open(train_tsv_path, 'r', encoding='utf-8')
    dev_tsv = open(dev_tsv_path, 'r', encoding='utf-8')
    test_tsv = open(test_tsv_path, 'r', encoding='utf-8')

    tsv_to_lst(args, train_tsv, args.dst + 'train_stratified.lst', data_partition='train')
    tsv_to_lst(args, dev_tsv, args.dst + 'dev_stratified.lst', data_partition='dev')
    tsv_to_lst(args, test_tsv, args.dst + 'test_stratified.lst', data_partition='test')

def get_filelines(file):
    lines = file.readlines()
    lines = [x.strip() for x in lines]
    return lines

def get_name2id_dict(file):
    name2id_dict = {}
    lines = get_filelines(file)
    for line in lines:
        line_split = line.split()
        name2id_dict[line_split[0]] = line_split[1]
    return name2id_dict

def normalize_transcript(transcript):
    # Remove punctuation signs.
    normalized_transcript = transcript.translate(str.maketrans('', '', string.punctuation))
    # Format to lower case.
    normalized_transcript = normalized_transcript.lower()
    # Remove exclamation and question marks.
    normalized_transcript = normalized_transcript.replace('!', '').replace('¡', '').replace('?', '').replace('¿', '')
    # Remove unnecessary whitespaces.
    normalized_transcript = normalized_transcript.replace('   ', ' ').replace('  ', ' ')
    return normalized_transcript

def tsv_to_lst(args, file, dst_path, data_partition='train'):
    sample_id = 0
    lines = get_filelines(file)[1:]
    with open(dst_path, "w", encoding='utf-8') as f:
        for line in lines:
            split_line = line.split('\t')
            file_path = split_line[1]
            if '.mp3' in file_path:
                file_path = file_path.replace('.mp3', args.audio_format)
            audio_path = args.docker_path + file_path
            if not os.path.isfile(audio_path):
                print("Missing: ")
                print(audio_path)
                continue
            audio_size = os.path.getsize(audio_path)
            if audio_size == 0:
                print(audio_path)
                continue
            audio_length = str(sox.file_info.duration(audio_path))
            transcript = split_line[2]
            transcript = normalize_transcript(transcript)

            writeline = []
            writeline.append(data_partition + str(sample_id))
            writeline.append(audio_path)
            writeline.append(audio_length)
            writeline.append(transcript)
            f.write("\t".join(writeline) + "\n")

            sample_id += 1

parser = argparse.ArgumentParser(description="CommonVoice Dataset creation.") 
parser.add_argument("--src", help="source directory", default="./")
parser.add_argument("--dst", help="destination directory", default="./")
parser.add_argument("--docker_path", help="path to data within docker", default="/ASR/data/CommonVoice/clips/")
parser.add_argument("--audio_format", help="audio file format", default=".wav")
parser.add_argument("--partition_type", help="type of partition, Common Voice proposal set by default: default | equivalidated", default="equivalidated")
#parser.add_argument("--partition_type", help="type of partition, Common Voice proposal set by default: default | equivalidated", default="default")
parser.add_argument('--max_downvotes', type=int, default=0, help='discard all recordings with more than the set maximum down vote number')
parser.add_argument('--stratification_threshold', type=float, default=18, help='allowed percentage margin for stratification')
parser.add_argument('--stratification_patience', type=float, default=1000, help='allowed iterations for finding a correct stratification')

args = parser.parse_args()
print("Arguments")
print(args)

assert os.path.isdir(str(args.src)), "CommonVoice src directory not found - '{d}'".format(d=args.src)

if args.partition_type == 'default':
    get_default_partition(args)
elif args.partition_type == 'equivalidated':
    get_equivalidated_partition(args)
