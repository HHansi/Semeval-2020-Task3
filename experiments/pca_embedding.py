# Created by Hansi at 2/9/2020
import numpy as np
from sklearn.decomposition import PCA

from algo.embeddings import get_embeddings, get_word_embeddings


def get_pca_embeddings(text, model, n):
    embeddings, words = get_embeddings(text, model)
    n_max = min(len(embeddings), len(embeddings[0]))
    if n > n_max:
        n = n_max

    pca = PCA(n_components=n)
    pca_embeddings = pca.fit_transform(np.array(embeddings))

    return pca_embeddings, words, n


# return word_vectors - [[[context1_word1_vec, context1_word2_vec], [context2_word1_vec, context_word2_vec]], []]
def get_pca_word_vectors(data, embedding_model, n):
    word_vectors = []
    for row in data:
        context1 = row[0]
        context2 = row[1]
        word1_context1 = row[2]
        word2_context1 = row[3]
        word1_context2 = row[4]
        word2_context2 = row[5]

        if len(context1) < len(context2):
            # context 1
            embeddings1, words1, n1 = get_pca_embeddings(context1, embedding_model, n)
            context1_word1_vec, context1_word2_vec = get_word_embeddings(words1, embeddings1, word1_context1,
                                                                         word2_context1)
            context1_vectors = [context1_word1_vec, context1_word2_vec]

            # context 2
            embeddings2, words2, n2 = get_pca_embeddings(context1, embedding_model, n1)
            context2_word1_vec, context2_word2_vec = get_word_embeddings(words2, embeddings2, word1_context2,
                                                                         word2_context2)
            context2_vectors = [context2_word1_vec, context2_word2_vec]
        else:
            # context 2
            embeddings2, words2, n2 = get_pca_embeddings(context1, embedding_model, n)
            context2_word1_vec, context2_word2_vec = get_word_embeddings(words2, embeddings2, word1_context2,
                                                                         word2_context2)
            context2_vectors = [context2_word1_vec, context2_word2_vec]

            # context 1
            embeddings1, words1, n1 = get_pca_embeddings(context1, embedding_model, n2)
            context1_word1_vec, context1_word2_vec = get_word_embeddings(words1, embeddings1, word1_context1,
                                                                         word2_context1)
            context1_vectors = [context1_word1_vec, context1_word2_vec]

        word_vectors.append([context1_vectors, context2_vectors])
    return word_vectors
