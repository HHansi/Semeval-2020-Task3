# Created by Hansi at 2/9/2020

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
    text = text.replace("<strong>", "")
    text = text.replace("</strong>", "")
    return text


def preprocessing_flow1(text):
    text = remove_additional_tags(text)
    text = clean_text1(text)
    return text
