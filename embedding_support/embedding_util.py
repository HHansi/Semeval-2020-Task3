# Created by Hansi at 2/23/2020
import sys

from unidecode import unidecode


def get_word_embeddings(tokens, embeddings, word1, word2):
    word1_embedding = []
    word2_embedding = []
    i = 0
    for token, embedding in zip(tokens, embeddings):
        if unidecode(token.lower()) == unidecode(word1.lower()):
            word1_embedding = embedding
            i += 1
            break

    for token, embedding in zip(tokens, embeddings):
        if unidecode(token.lower()) == unidecode(word2.lower()):
            word2_embedding = embedding
            i += 1
            break

    if i != 2:
        print(f'i = {i}')
        print(f'Word1 = {unidecode(word1.lower())}')
        print(f'Word2 = {unidecode(word2.lower())}')
        print(tokens)
        sys.exit(f'Error: Not able to find the words in context')

    return word1_embedding, word2_embedding

# get average of given embeddings
def get_embedding_average(emb1, emb2, chuck_size):
    emb1_splits = np.split(emb1, chuck_size)
    emb2_splits = np.split(emb2, chuck_size)
    emb1_sum = np.zeros(shape=len(emb1_splits[0]))
    for emb in emb1_splits:
        emb1_sum = emb1_sum + emb
    emb2_sum = np.zeros(shape=len(emb2_splits[0]))
    for emb in emb2_splits:
        emb2_sum = emb2_sum + emb
    emb1_avg = emb1_sum / chuck_size
    emb2_avg = emb2_sum / chuck_size
    return emb1_avg, emb2_avg
