# Created by Hansi at 2/9/2020
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, StackedEmbeddings


def get_sentence_embeddings(text, model):
    sentence = Sentence(text)
    model.embed(sentence)
    return sentence


def get_embeddings(text, model):
    embeddings = []
    words = []
    sentence = Sentence(text)
    model.embed(sentence)
    for token in sentence:
        words.append(token.text)
        embeddings.append(token.embedding.numpy())
    return embeddings, words


def get_word_embeddings(words, embeddings, word1, word2):
    word1_embedding = []
    word2_embedding = []

    i = 0
    for word in words:
        if word.lower() == word1.lower():
            word1_embedding = embeddings[i]
        elif word.lower() == word2.lower():
            word2_embedding = embeddings[i]
        i += 1

        if len(word1_embedding) != 0 and len(word2_embedding) != 0:
            break
    return word1_embedding, word2_embedding


def get_bert(model_name):
    return BertEmbeddings(model_name)


def get_elmo(model_name):
    return ELMoEmbeddings(model_name)


def get_stacked_embeddings(model1, model2):
    # create the StackedEmbedding object that combines all embeddings
    stacked_embeddings = StackedEmbeddings(
        embeddings=[model1, model2])
    return stacked_embeddings

