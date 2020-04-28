# Created by Hansi at 2/9/2020
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, StackedEmbeddings, XLNetEmbeddings, \
    TransformerXLEmbeddings, FlairEmbeddings, XLMRobertaEmbeddings, XLMEmbeddings, OpenAIGPT2Embeddings


def get_embeddings_flair(text, model):
    embeddings = []
    words = []
    sentence = Sentence(text)
    model.embed(sentence)
    for token in sentence:
        words.append(token.text)
        embeddings.append(token.embedding.numpy())
    return embeddings, words


def get_bert(model_name, layers='-1,-2,-3,-4', pooling_op='first', scalar_mix=False):
    return BertEmbeddings(model_name, layers=layers, pooling_operation=pooling_op, use_scalar_mix=scalar_mix)


def get_elmo(model_name):
    return ELMoEmbeddings(model_name)


def get_xlnet(model_name):
    return XLNetEmbeddings(model_name)


def get_transformerxl(model_name):
    return TransformerXLEmbeddings(model_name)


def get_flair(model_name):
    return FlairEmbeddings(model_name)


def get_xlm(model_name):
    return XLMEmbeddings(model_name)


def get_xml_roberta(model_name):
    return XLMRobertaEmbeddings(model_name)


def get_gpt2(model_name):
    return OpenAIGPT2Embeddings()


def get_stacked_embedding_model(models):
    stacked_embedding_model = StackedEmbeddings(
        embeddings=models)
    return stacked_embedding_model


