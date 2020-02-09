# Created by Hansi at 2/9/2020
import os

import pandas as pd

from data_processing.preprocessor import preprocessing_flow1


def load_raw_data(file_path):
    data = pd.read_csv(file_path, sep='\t', engine='python', encoding='utf-8')
    return data


def create_folder_if_not_exist(path, is_file_path=False):
    if is_file_path:
        folder_path = '/'.join(path.split('/')[0:-1])
    else:
        folder_path = path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# return data as array of rows [context1, context2, word1_context1, word2_context1, word1_context2, word2_context2]
def load_structured_data(file_path, language, preprocess=False):
    structured_data = []
    data = load_raw_data(file_path)
    for index, row in data.iterrows():
        if language == 'en':
            word1_context1 = row['word1']
            word2_context1 = row['word2']
            word1_context2 = row['word1']
            word2_context2 = row['word2']
        else:
            word1_context1 = row['word1_context1']
            word2_context1 = row['word2_context1']
            word1_context2 = row['word1_context2']
            word2_context2 = row['word2_context2']

        context1 = row['context1']
        context2 = row['context2']
        if preprocess:
            context1 = preprocessing_flow1(context1)
            context2 = preprocessing_flow1(context2)
        structured_data.append([context1, context2, word1_context1, word2_context1, word1_context2, word2_context2])
    return structured_data
