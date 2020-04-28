# Created by Hansi at 2/23/2020
import sys

from unidecode import unidecode


# tokens - list of all tokens/words
# embeddings - list of all embeddings correspond to the tokens
# word1, word2 - words for which embeddings need to be extracted
# return embedding for word1 and embedding for word2
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
