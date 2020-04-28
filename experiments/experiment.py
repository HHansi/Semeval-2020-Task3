# Created by Hansi at 2/23/2020
import csv

from data_processing.data_preprocessor import remove_additional_tags
from embedding_support.average_embedding import get_average_embeddings
from embedding_support.embedding import get_embeddings_flair, get_bert
from embedding_support.embedding_util import get_word_embeddings
from evaluation.evaluation1 import evaluate1
from evaluation.evaluation2 import evaluate2
from experiments.result_generator import write_results
from experiments.similarity_measures import get_cosine_similarity
from project_config import data_folder, result_folder


def generate_results(input_folder_path, output_folder_path, dict_language_model=None, n=None):
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
                context1 = remove_additional_tags(context1)
                context2 = remove_additional_tags(context2)

                # context1 calculation
                # embeddings, tokens = get_embeddings_baseline(context1)
                embeddings, tokens = get_embeddings_flair(context1, dict_language_model[lan])
                word1_embedding, word2_embedding = get_word_embeddings(tokens, embeddings, row['word1_context1'],
                                                                       row['word2_context1'])
                if n:
                    word1_embedding, word2_embedding = get_average_embeddings(word1_embedding, word2_embedding, n)
                    print('average embeddings are generated for context1')
                sim_context1 = get_cosine_similarity(word1_embedding, word2_embedding)

                # context2 calculation
                # embeddings, tokens = get_embeddings_baseline(context2)
                embeddings, tokens = get_embeddings_flair(context2, dict_language_model[lan])
                word1_embedding, word2_embedding = get_word_embeddings(tokens, embeddings, row['word1_context2'],
                                                                       row['word2_context2'])
                if n:
                    word1_embedding, word2_embedding = get_average_embeddings(word1_embedding, word2_embedding, n)
                    print('average embeddings context2')
                sim_context2 = get_cosine_similarity(word1_embedding, word2_embedding)

                similarities.append([sim_context1, sim_context2])

        write_results(similarities, output_folder_path, lan)


# input_folder = data_folder + 'evaluation_kit_final/data/'
input_folder = data_folder + 'practice_kit_final/data/'
output_folder = result_folder + 'final-results/experiment1/'

languages = ('en', 'hr', 'sl')
lang_emb = dict()
model = get_bert('bert-base-multilingual-uncased')
lang_emb['en'] = model
lang_emb['hr'] = model
lang_emb['sl'] = model
# lang_emb['fi'] = model

generate_results(input_folder, output_folder, lang_emb)

# evaluate results
gold_folder = data_folder + 'practice_kit_final/gold'
print('Subtask 1')
evaluate1(gold_folder, output_folder + '/res1/')

print('Subtask 2')
evaluate2(gold_folder, output_folder + '/res2/')
