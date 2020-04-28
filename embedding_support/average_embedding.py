# Created by Hansi at 2/23/2020
import numpy as np


# embedding - concatenated embedding extracted by model for n layers
# return average embedding
def get_average_embeddings(embedding, n):
    emb_splits = np.split(embedding, n)
    emb_sum = np.zeros(shape=len(emb_splits[0]))
    for emb in emb_splits:
        emb_sum = emb_sum + emb

    emb_avg = emb_sum / n

    return emb_avg
