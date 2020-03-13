# Created by Hansi at 2/23/2020
from bert_embedding import BertEmbedding

## Load bert model
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_uncased', max_seq_length=230)


def get_embeddings_baseline(text, model=None):
    ## Results contains several results for several sentences
    results = bert_embedding([text])
    ## Result here contains the results for first sentence
    result = results[0]
    ## result[0] is a list with all the tokens
    ## result[1] is a list with the embeddings per token
    tokens = result[0]
    embeddings = result[1]
    return embeddings, tokens
