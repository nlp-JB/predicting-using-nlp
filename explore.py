import numpy as np
import pandas as pd

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

import j_acquire
import j_prep

def make_initial_df(text_file):
    data = j_acquire.scrape_github_data(text_file)
    df = pd.DataFrame.from_dict(data)
    df = df.dropna().reset_index()
    df = j_prep.prep_readme_data(df)
    return df

def add_new_columns(df):
    # add generalized language column
    df['gen_language'] = np.where(df.language == 'Python', 'Python',
                             np.where(df.language == 'JavaScript', 'JavaScript',
                                     np.where(df.language == 'Jupyter Notebook', 'Jupyter Notebook', 'other')))
    # Add column of text without numbers
    df['without_numbers']= df['clean_lemmatized'].str.lower().replace(r'(?<![A-Za-z0-9])[0-9]+', '')
    # turn word strings into a list of words
    tokenizer = nltk.tokenize.ToktokTokenizer()
    df['num_words'] = df.without_numbers.apply(lambda x: len(tokenizer.tokenize(x)))
    df['num_unique_words'] = df.without_numbers.apply(lambda x: len(set(tokenizer.tokenize(x))))
    # Count the number of links in each readme
    df['link_counts'] = df.readme_contents.str.count(r'https*')
    # Add column with count of .py files
    df['py_extensions'] = df.clean_lemmatized.str.count(r'py\b')
    # Add column with count of .js files
    df['js_extensions'] = df.clean_lemmatized.str.count(r'js\b')
    # Add column with count of .ipynb files
    df['ipynb_extensions'] = df.clean_lemmatized.str.count(r'ipynb\b')
    
    return df
    
    
def make_word_counts_df(df):
    # Turn all words from each category into a single string
    python_words = ' '.join(df[df.language == 'Python'].without_numbers)
    java_words = ' '.join(df[df.language == 'JavaScript'].without_numbers)
    jupyter_words = ' '.join(df[df.language == 'Jupyter Notebook'].without_numbers)
    other_words = ' '.join(df[(df.language != 'Jupyter Notebook') 
                              & (df.language != 'Python') & (df.language != 'JavaScript')].without_numbers)
    all_words = ' '.join(df.without_numbers)
    
    # tokenize the words, then count them
    tokenizer = nltk.tokenize.ToktokTokenizer()

    python_words_freq = pd.Series(tokenizer.tokenize(python_words)).value_counts()
    java_words_freq = pd.Series(tokenizer.tokenize(java_words)).value_counts()
    jupyter_words_freq = pd.Series(tokenizer.tokenize(jupyter_words)).value_counts()
    other_words_freq = pd.Series(tokenizer.tokenize(other_words)).value_counts()
    all_words_freq = pd.Series(tokenizer.tokenize(all_words)).value_counts()
    
    # combine them into a word_count_df
    word_counts = (pd.concat([all_words_freq, python_words_freq, java_words_freq, 
                          jupyter_words_freq, other_words_freq], axis=1, sort=True)
              .set_axis(['all', 'python', 'java', 'jupyter', 'other'], axis=1, inplace=False)
              .fillna(0)
              .apply(lambda s: s.astype(int)))
    
    return word_counts

def make_vectorized_df(df):
    tfidf = TfidfVectorizer()
    tfidfs = tfidf.fit_transform(df.without_numbers)
    vectorized_df = pd.DataFrame(tfidfs.todense(), columns=tfidf.get_feature_names())
    # add caluclulated features to vectorized_df
    vectorized_df = vectorized_df.join(df[[
        'num_words', 'num_unique_words', 'link_counts', 'py_extensions',
        'js_extensions', 'ipynb_extensions']], how='left')
    return vectorized_df