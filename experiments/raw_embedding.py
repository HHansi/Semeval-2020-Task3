# Created by Hansi at 2/9/2020

from algo.embeddings import get_word_embeddings, get_embeddings


# return word_vectors - [[[context1_word1_vec, context1_word2_vec], [context2_word1_vec, context_word2_vec]], []]
def get_word_vectors(data, embedding_model):
    word_vectors = []
    for row in data:
        context1 = row[0]
        context2 = row[1]
        word1_context1 = row[2]
        word2_context1 = row[3]
        word1_context2 = row[4]
        word2_context2 = row[5]

        # context 1
        embeddings1, words1 = get_embeddings(context1, embedding_model)
        context1_word1_vec, context1_word2_vec = get_word_embeddings(words1, embeddings1, word1_context1,
                                                                         word2_context1)
        context1_vectors = [context1_word1_vec, context1_word2_vec]

        # context 2
        embeddings2, words2 = get_embeddings(context2, embedding_model)
        context2_word1_vec, context2_word2_vec = get_word_embeddings(words2, embeddings2, word1_context2,
                                                                         word2_context2)
        context2_vectors = [context2_word1_vec, context2_word2_vec]

        word_vectors.append([context1_vectors, context2_vectors])
    return word_vectors
