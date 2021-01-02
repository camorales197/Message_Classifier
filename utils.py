import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def read_files(category):
    """
    Input: State a category to pick all the messages from
    Output: A dataset with two columns: text of the message, and a tag of the category associated to the message
    """
    files_content_list = []
    category_path = os.path.join('dataset', category)
    files_names_list = os.listdir(category_path)
    for file_name in files_names_list:
        file_path = os.path.join(category_path, file_name)
        content = open(file_path, 'r', errors='ignore').read()
        files_content_list.append(content)
        df_files_content = pd.DataFrame(files_content_list, columns=['Text'])
        df_files_content['Category'] = category
    return df_files_content


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        """
        Text preprocessing transformer to clean the messages
        """

    def fit(self, X, y=None):
        return self  # returns the same

    def transform(self, X, y=None):
        X_transform = X.map(lambda s: self.__preprocess(s))
        return X_transform

    def __preprocess(self, sentence):
        """
        Input: String of text
        Output: String of text with only words and not other characters
        """
        import re
        from nltk.tokenize import RegexpTokenizer
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        # from spellchecker import SpellChecker #It takes way too long (>1h)

        # self.spell = SpellChecker() #It takes way too long (>1h)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        rem__ = re.sub('_+', '', rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem__)
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        # corrected_words = [self.spell.correction(w) for w in filtered_words] #It takes way too long (>1h)
        stem_words = [self.stemmer.stem(w) for w in filtered_words]
        lemma_words = [self.lemmatizer.lemmatize(w) for w in stem_words]
        return " ".join(lemma_words)