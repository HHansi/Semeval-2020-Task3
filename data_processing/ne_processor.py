# Created by Hansi at 2/28/2020
import csv

import spacy

from data_processing.data_preprocessor import preprocessing_flow
from data_processing.file_util import create_folder_if_not_exist
from project_config import model_folder, data_folder

nlp = spacy.load('en_core_web_sm')

ignored_entities = ['NORP', 'WORK_OF_ART', 'DATE', 'TIME', 'QUANTITY', 'ORDINAL', 'CARDINAL']

entity_names = dict()
entity_names['PERSON'] = 'person'
entity_names['NORP'] = 'nationality'
entity_names['FAC'] = 'building'
entity_names['ORG'] = 'organisation'
entity_names['GPE'] = 'country'
entity_names['LOC'] = 'location'
entity_names['PRODUCT'] = 'object'
entity_names['EVENT'] = 'event'
entity_names['WORK_OF_ART'] = 'title'
entity_names['LAW'] = 'law'
entity_names['LANGUAGE'] = 'language'
entity_names['DATE'] = 'date'
entity_names['TIME'] = 'time'
entity_names['PERCENT'] = 'percentage'
entity_names['MONE'] = 'money'
entity_names['QUANTITY'] = 'quantity'
entity_names['ORDINAL'] = 'ordinal'  # not in vocab
entity_names['CARDINAL'] = 'cardinal'


# Read vocabulary of trained model
def read_vocab(filepath):
    f = open(filepath, 'r', encoding='utf-8')
    vocab = f.readlines()
    f.close()
    vocab = [str.replace(word, '\n', '') for word in vocab]
    return vocab


# Filter unknown words to the vocabulary
# vocab - vocabulary as a list of words
# words - word list to compare
def get_unknown_words(vocab, words):
    unknowns = set()
    for word in words:
        if word not in vocab:
            unknowns.add(word)
    return unknowns


# Identify named entities in the given string
def get_entities(str):
    dict_entities = dict()
    doc = nlp(str)
    for ent in doc.ents:
        dict_entities[ent.text] = ent.label_
    print([(X.text, X.label_) for X in doc.ents])
    return dict_entities


# Replace unknown words in the context by known entities
def replace_with_entities(context, unknown_words):
    replaced_words = []
    new_context = context
    dict_entities = get_entities(context)
    for word in unknown_words:
        if word in dict_entities.keys():

            if dict_entities[word] not in ignored_entities:
                new_context = str.replace(new_context, word, entity_names[dict_entities[word]])
                replaced_words.append(word + '-' + entity_names[dict_entities[word]])
    return new_context, replaced_words


# generate new data set by replacing unknown tokens by known named entities
# file_path - path to data file
# vocab_path - path to vocabulary (BERT models have separate .txt file which contains its vocabulary)
# output_filepath_diff - .tsv file path to write difference between old context and nex context (only for analysis
# purpose)
# output_filepath_data - .tsv file path to write final data set
def replace_with_entities_bulk(file_path, vocab_path, output_filepath_diff, output_filepath_data):
    # read vocabulary of trained model
    vocab = read_vocab(vocab_path)

    # open file to write words which are not in the vocabulary (for analysis purpose)
    create_folder_if_not_exist(output_filepath_diff, is_file_path=True)
    csv_file_out = open(output_filepath_diff, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file_out, delimiter='\t')

    # open file to write entity replaced data
    create_folder_if_not_exist(output_filepath_data, is_file_path=True)
    csv_file_out_data = open(output_filepath_data, 'a', newline='', encoding='utf-8')
    csv_writer_data = csv.writer(csv_file_out_data, delimiter='\t')

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for index, row in enumerate(csvreader):
            print(f"{index} {row['word1']}-{row['word2']}")

            context1 = row['context1']
            context2 = row['context2']

            # preprocess text; context1
            context1 = preprocessing_flow(context1)
            # get words in context1
            context1_words = context1.split()
            # identify words which are not in vocabulary
            unknown_words1 = get_unknown_words(vocab, context1_words)
            # replace unknown words in the context by named entities
            new_context1, replaced_words1 = replace_with_entities(row['context1'], unknown_words1)

            # preprocess text; context2
            context2 = preprocessing_flow(context2)
            # get words in context2
            context2_words = context2.split()
            # identify words which are not in vocabulary
            unknown_words2 = get_unknown_words(vocab, context2_words)
            # replace unknown words in the context by named entities
            new_context2, replaced_words2 = replace_with_entities(row['context2'], unknown_words2)

            csv_writer.writerow(
                [','.join(unknown_words1), ','.join(unknown_words2), ','.join(replaced_words1), ','.join(replaced_words2)])
            csv_writer_data.writerow(
                [row['word1'], row['word2'], new_context1, new_context2, row['word1_context1'], row['word2_context1'],
                 row['word1_context2'], row['word2_context2']])

    csv_file_out.close()
    csv_file_out_data.close()


if __name__ == '__main__':
    vocab_path = model_folder + 'bert/cased_L-24_H-1024_A-16/vocab.txt'
    file_path = data_folder + '/evaluation_kit_final/data/data_en.tsv'
    output_filepath_diff = data_folder + 'evaluation_kit_final/diff-sm/diff_en_without-ents.tsv'
    output_filepath_data = data_folder + 'evaluation_kit_final/diff-sm/data_en_without-ents.tsv'
    replace_with_entities_bulk(file_path, vocab_path, output_filepath_diff, output_filepath_data)