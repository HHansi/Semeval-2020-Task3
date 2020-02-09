# Created by Hansi at 2/9/2020
import csv
import os

from algo.embeddings import get_bert
from algo.similarity_measures import get_cosine_similarity

from data_processing.data_util import load_structured_data
from evaluation.evaluation1 import evaluate1
from evaluation.evaluation2 import evaluate2
from experiments.pca_embedding import get_pca_word_vectors
from experiments.raw_embedding import get_word_vectors


# context1_vectors - [word1_vec, word2_vec]
def get_similarity_measures(context1_vectors, context2_vectors):
    sim_context1 = get_cosine_similarity(context1_vectors[0], context1_vectors[1])
    sim_context2 = get_cosine_similarity(context2_vectors[0], context2_vectors[1])

    diff = sim_context2 - sim_context1
    return sim_context1, sim_context2, diff


def generate_results(languages, dict_lang_model, input_folder, output_folder):
    output_folder_subtask1 = output_folder + '/subtask1'
    output_folder_subtask2 = output_folder + '/subtask2'

    if not os.path.exists(output_folder_subtask1):
        os.makedirs(output_folder_subtask1)

    if not os.path.exists(output_folder_subtask2):
        os.makedirs(output_folder_subtask2)

    for lang in languages:
        input_filepath = input_folder + os.path.sep + 'data_' + lang + '.tsv'
        data = load_structured_data(input_filepath, lang, preprocess=True)

        # change vector generation for different experiments
        # word_vectors = get_word_vectors(data, dict_lang_model[lang])
        word_vectors = get_pca_word_vectors(data, dict_lang_model[lang], 50)

        task1_output_filepath = output_folder_subtask1 + os.path.sep + 'results_subtask1_' + lang + '.tsv'
        task2_output_filepath = output_folder_subtask2 + os.path.sep + 'results_subtask2_' + lang + '.tsv'

        csv_file_out_task1 = open(task1_output_filepath, 'a', newline='', encoding='utf-8')
        csv_writer_task1 = csv.writer(csv_file_out_task1, delimiter='\t')
        csv_writer_task1.writerow(['change'])

        csv_file_out_task2 = open(task2_output_filepath, 'a', newline='', encoding='utf-8')
        csv_writer_task2 = csv.writer(csv_file_out_task2, delimiter='\t')
        csv_writer_task2.writerow(['sim_context1', 'sim_context2'])

        for row in word_vectors:
            context1_vectors = row[0]
            context2_vectors = row[1]

            sim_context1, sim_context2, diff = get_similarity_measures(context1_vectors, context2_vectors)
            csv_writer_task2.writerow([sim_context1, sim_context2])

            diff = sim_context2 - sim_context1
            csv_writer_task1.writerow([diff])

        csv_file_out_task1.close()
        csv_file_out_task2.close()


if __name__ == '__main__':
    languages = ('en', 'hr', 'sl')
    lang_emb = dict()
    lang_emb['en'] = get_bert('bert-large-cased')
    bert_multilingual = get_bert('bert-base-multilingual-cased')
    lang_emb['hr'] = bert_multilingual
    lang_emb['sl'] = bert_multilingual

    input_folder = '../data/trial_data_en_hr_sl/trial_data_task3/data'
    output_folder = '../results/bert_embedding_large_cased_en_multilingual_cased-pca'

    generate_results(languages, lang_emb, input_folder, output_folder)

    gold_folder = '../data/trial_data_en_hr_sl/trial_data_task3/gold'
    print('Sub task 1')
    evaluate1(gold_folder, output_folder + '/subtask1')

    print('Sub task 2')
    evaluate2(gold_folder, output_folder + '/subtask2')
