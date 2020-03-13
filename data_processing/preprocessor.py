# Created by Hansi at 2/9/2020
import csv

import spacy

from project_config import model_folder, data_folder

nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_lg')

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', ]

# ignored_entities = ['NORP', 'WORK_OF_ART', 'LAW', 'ORDINAL'] # entsV1
ignored_entities = ['NORP', 'WORK_OF_ART', 'DATE', 'TIME', 'QUANTITY', 'ORDINAL', 'CARDINAL']  # entsV2
# ignored_entities = []

entity_names = dict()
entity_names['PERSON'] = 'person'
entity_names['NORP'] = 'nationality'
entity_names['FAC'] = 'building'
entity_names['ORG'] = 'organisation'
entity_names['GPE'] = 'country'
entity_names['LOC'] = 'location'
entity_names['PRODUCT'] = 'object'
entity_names['EVENT'] = 'event'
entity_names['WORK_OF_ART'] = 'title'
entity_names['LAW'] = 'law'
entity_names['LANGUAGE'] = 'language'
entity_names['DATE'] = 'date'
entity_names['TIME'] = 'time'
entity_names['PERCENT'] = 'percentage'
entity_names['MONE'] = 'money'
entity_names['QUANTITY'] = 'quantity'
entity_names['ORDINAL'] = 'ordinal'  # not in vocab
entity_names['CARDINAL'] = 'cardinal'


# Add white space before and after each punctuation mark
def clean_text1(x):
    #     x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


# Remove punctuation marks
def clean_text2(x):
    #     x = str(x)
    for punct in puncts:
        x = x.replace(punct, ' ')
    return x


# Remove additional tags
def remove_additional_tags(text):
    text = text.replace("<strong>", " ")
    text = text.replace("</strong>", " ")
    return text


def unbolded(context):
    #     context = str.replace(context, '-', '')
    context = str.replace(context, '<strong>', '')
    return str.replace(context, '</strong>', '')


def preprocessing_flow1(text):
    text = remove_additional_tags(text)
    return text


def preprocessing_flow2(text):
    text = remove_additional_tags(text)
    text = clean_text1(text)
    return text


def read_vocab(filepath):
    f = open(filepath, 'r', encoding='utf-8')
    x = f.readlines()
    f.close()
    x = [str.replace(word, '\n', '') for word in x]
    return x


def get_word_diff(vocab, words):
    diff = set()
    for word in words:
        if word not in vocab:
            diff.add(word)
    return diff


def get_entities(str):
    dict_entities = dict()
    doc = nlp(str)
    for ent in doc.ents:
        print(ent.text)
        dict_entities[ent.text] = ent.label_
    print([(X.text, X.label_) for X in doc.ents])
    return dict_entities


def replace_with_entities(context, diff_words):
    replaced_words = []
    new_context = context
    dict_entities = get_entities(context)
    for word in diff_words:
        if word in dict_entities.keys():
            print(word)

            if dict_entities[word] not in ignored_entities:
                new_context = str.replace(new_context, word, entity_names[dict_entities[word]])
                replaced_words.append(word + '-' + entity_names[dict_entities[word]])
    print(replaced_words)
    return new_context, replaced_words


def replace_with_entities_bulk(file_path, vocab_path, output_filepath_diff, output_filepath_data):
    vocab = read_vocab(vocab_path)
    csv_file_out = open(output_filepath_diff, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file_out, delimiter='\t')

    csv_file_out_data = open(output_filepath_data, 'a', newline='', encoding='utf-8')
    csv_writer_data = csv.writer(csv_file_out_data, delimiter='\t')

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for index, row in enumerate(csvreader):
            print(f"{index} {row['word1']}-{row['word2']}")

            context1 = row['context1']
            context2 = row['context2']
            # preprocess text
            context1 = preprocessing_flow2(context1)
            context1_words = context1.split()
            new_words1 = get_word_diff(vocab, context1_words)
            new_context1, replaced_words1 = replace_with_entities(row['context1'], new_words1)

            context2 = preprocessing_flow2(context2)
            context2_words = context2.split()
            new_words2 = get_word_diff(vocab, context2_words)
            new_context2, replaced_words2 = replace_with_entities(row['context2'], new_words2)

            csv_writer.writerow(
                [','.join(new_words1), ','.join(new_words2), ','.join(replaced_words1), ','.join(replaced_words2)])
            csv_writer_data.writerow(
                [row['word1'], row['word2'], new_context1, new_context2, row['word1_context1'], row['word2_context1'],
                 row['word1_context2'], row['word2_context2']])

    csv_file_out.close()


if __name__ == '__main__':
    vocab_path = model_folder + 'bert/cased_L-24_H-1024_A-16/vocab.txt'
    file_path = data_folder + '/evaluation_kit_final/data/data_en.tsv'
    output_filepath_diff = data_folder + 'evaluation_kit_final/diff-sm/diff_en_without-ents.tsv'
    output_filepath_data = data_folder + 'evaluation_kit_final/diff-sm/data_en_without-ents.tsv'
    replace_with_entities_bulk(file_path, vocab_path, output_filepath_diff, output_filepath_data)
