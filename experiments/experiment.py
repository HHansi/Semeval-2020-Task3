# Created by Hansi at 2/23/2020

# Experiment updated based on the baseline code provided

import csv

from flair.embeddings import BertEmbeddings

from embedding_support.embedding_util import get_word_embeddings, get_embedding_average
from embedding_support.flair_embeddings import get_embeddings_flair
from embedding_support.similarity_measures import get_cosine_similarity
from data_processing.data_util import create_folder_if_not_exist
from data_processing.preprocessor import preprocessing_flow1
from evaluation.evaluation1 import evaluate1
from evaluation.evaluation2 import evaluate2
from project_config import result_folder, data_folder

languages = ['en']
# languages = ('en', 'fi', 'hr', 'sl')


def generate_results(input_folder_path, output_folder_path, dict_language_model=None, chunk_size=None):
    for lan in languages:
        rows = []
        similarities = []

        print(f'\nLANGUAGE: {lan.upper()}\n')

        with open(input_folder_path + f'/data_{lan}.tsv', 'r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for index, row in enumerate(csvreader):
                print(f"{index} {row['word1']}-{row['word2']}")

                context1 = row['context1']
                context2 = row['context2']
                # preprocess text
                context1 = preprocessing_flow1(context1)
                context2 = preprocessing_flow1(context2)

                # context1 calculation
                # embeddings, tokens = get_embeddings_baseline(context1)
                embeddings, tokens = get_embeddings_flair(context1, dict_language_model[lan])
                word1_embedding, word2_embedding = get_word_embeddings(tokens, embeddings, row['word1_context1'],
                                                                       row['word2_context1'])
                if chunk_size:
                    word1_embedding, word2_embedding = get_embedding_average(word1_embedding, word2_embedding,
                                                                             chunk_size)
                    print('average embeddings context1')
                sim_context1 = get_cosine_similarity(word1_embedding, word2_embedding)

                # context2 calculation
                # embeddings, tokens = get_embeddings_baseline(context2)
                embeddings, tokens = get_embeddings_flair(context2, dict_language_model[lan])
                word1_embedding, word2_embedding = get_word_embeddings(tokens, embeddings, row['word1_context2'],
                                                                       row['word2_context2'])
                if chunk_size:
                    word1_embedding, word2_embedding = get_embedding_average(word1_embedding, word2_embedding,
                                                                             chunk_size)
                    print('average embeddings context2')
                sim_context2 = get_cosine_similarity(word1_embedding, word2_embedding)

                similarities.append([sim_context1, sim_context2])

        columns = ['change']
        with open(output_folder_path + f'/res1/results_subtask1_{lan}.tsv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
            csvwriter.writerow(columns)
            for sim in similarities:
                change = sim[1] - sim[0]
                csvwriter.writerow([change])

        columns = ['sim_context1', 'sim_context2']
        with open(output_folder_path + f'/res2/results_subtask2_{lan}.tsv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
            csvwriter.writerow(columns)
            for sim in similarities:
                csvwriter.writerow(sim)


if __name__ == '__main__':
    input_folder = '../data/evaluation_kit_final/diff-sm/ignored-ents-V2/'
    output_folder = result_folder + 'final-results/submission6/bert-large-cased-layers-1-14-avg_sm-ignored-entsv2/'

    # input_folder = data_folder + 'trial_data_en_hr_sl/trial_data_task3/data/'
    # input_folder = data_folder + 'practice_kit_final/diff-sm/ignored-ents-V2/'
    # output_folder = result_folder + 'practice-kit-final/English3/bert-large-cased-layers-1-14-avg_sm-ignored-entsv2/'

    create_folder_if_not_exist(output_folder)
    res1_folder = output_folder + 'res1/'
    create_folder_if_not_exist(res1_folder)
    res2_folder = output_folder + 'res2/'
    create_folder_if_not_exist(res2_folder)

    lang_emb = dict()
    model = BertEmbeddings('bert-large-cased', layers='-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14')
    # model = BertEmbeddings('bert-base-finnish-uncased-v1', pooling_operation='first_last', use_scalar_mix=True)

    # model2 = get_flair('sl-backward')
    # models = [model1, model2]
    # model = get_stacked_embeddings(models)

    lang_emb['en'] = model
    # lang_emb['hr'] = model
    # lang_emb['sl'] = model
    # lang_emb['fi'] = model
    generate_results(input_folder, output_folder, lang_emb, chunk_size=14)

    # evaluate results
    output_folder = result_folder + 'practice-kit-final/English3/bert-large-cased-layers-1-14-avg_sm-ignored-entsv2/'
    # gold_folder = data_folder + 'trial_data_en_hr_sl/trial_data_task3/gold'
    gold_folder = data_folder + 'practice_kit_final/gold'
    print('Sub task 1')
    evaluate1(gold_folder, output_folder + '/res1/')

    print('Sub task 2')
    evaluate2(gold_folder, output_folder + '/res2/')
