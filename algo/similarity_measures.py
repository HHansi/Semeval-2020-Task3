# Created by Hansi at 2/9/2020
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_cosine_similarity(vec1, vec2):
    dimensions = len(vec1)
    vec11 = vec1.reshape(1, dimensions)
    vec22 = vec2.reshape(1, dimensions)
    return cosine_similarity(vec11, vec22)[0][0]


#     #Cosine similarity = dot product for normalized vectors
#     return dot(matutils.unitvec(vec1), matutils.unitvec(vec2))

def get_euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def get_euclidean_similarity(vec1, vec2):
    return 1 / (1 + get_euclidean_distance(vec1, vec2))
