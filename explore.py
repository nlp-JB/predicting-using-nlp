import numpy as np
import pandas as pd

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import j_acquire
import j_prep

def make_initial_df():
    text_files = ['algorithm', 'bots', 'data-visualization',
              'deep_learning', 'javascript', 'jupyter_notebook',
              'machine_learning', 'nlp', 'python',
              'repo_source', 'testing']
    
    df = j_acquire.create_large_df(text_files)
    df = j_prep.prep_readme_data(df)
    return df

def add_new_columns(df):
    # add generalized language column
    df['gen_language'] = np.where(df.language == 'Python', 'Python',
                             np.where(df.language == 'JavaScript', 'JavaScript',
                                     np.where(df.language == 'Jupyter Notebook', 'Jupyter Notebook',
                                              np.where(df.language == 'C++', 'C++',
                                                      np.where(df.language == 'TypeScript', 'TypeScript',
                                                               np.where(df.language == 'Java', 'Java', 'other'))))))
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
    javascript_words = ' '.join(df[df.language == 'JavaScript'].without_numbers)
    jupyter_words = ' '.join(df[df.language == 'Jupyter Notebook'].without_numbers)
    c_plus_words = ' '.join(df[df.language == 'C++'].without_numbers)
    typescript_words = ' '.join(df[df.language == 'TypeScript'].without_numbers)
    java_words = ' '.join(df[df.language == 'Java'].without_numbers)
    other_words = ' '.join(df[
        (df.language != 'Jupyter Notebook') 
        & (df.language != 'Python') 
        & (df.language != 'JavaScript')
        & (df.language != 'C++')
        & (df.language != 'TypeScript')
        & (df.language != 'Java')
                             ].without_numbers)
    all_words = ' '.join(df.without_numbers)
    
    # tokenize the words, then count them
    tokenizer = nltk.tokenize.ToktokTokenizer()

    python_words_freq = pd.Series(tokenizer.tokenize(python_words)).value_counts()
    javascript_words_freq = pd.Series(tokenizer.tokenize(javascript_words)).value_counts()
    jupyter_words_freq = pd.Series(tokenizer.tokenize(jupyter_words)).value_counts()
    c_plus_words_freq = pd.Series(tokenizer.tokenize(c_plus_words)).value_counts()
    typescript_words_freq = pd.Series(tokenizer.tokenize(typescript_words)).value_counts()
    java_words_freq = pd.Series(tokenizer.tokenize(java_words)).value_counts()
    other_words_freq = pd.Series(tokenizer.tokenize(other_words)).value_counts()
    all_words_freq = pd.Series(tokenizer.tokenize(all_words)).value_counts()
    
    # combine them into a word_count_df
    word_counts = (pd.concat([all_words_freq, python_words_freq, javascript_words_freq, 
                              jupyter_words_freq, c_plus_words_freq, typescript_words_freq,
                              java_words_freq, other_words_freq], axis=1, sort=True)
              .set_axis(['all', 'python', 'javascript', 'jupyter', 'c_plus',
                         'typescript', 'java', 'other'], axis=1, inplace=False)
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

def min_max_scaler(train, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm

    This is a linear transformation. Values will lie between 0 and 1
    '''
    mm_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled_mm = pd.DataFrame(mm_scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled_mm = pd.DataFrame(mm_scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return mm_scaler, train_scaled_mm, test_scaled_mm


def get_splits(df, vectorized_df):
    X = vectorized_df
    y = df.gen_language
    #Split dataframe into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state = 123)
    train_predictions = pd.DataFrame(dict(actual=y_train))
    test_predictions = pd.DataFrame(dict(actual=y_test))

    # Scale data
    scaler, X_train_scaled, X_test_scaled = min_max_scaler(X_train, X_test)

    return scaler, X_train_scaled, X_test_scaled, y_train, y_test, train_predictions, test_predictions

def prep_vectorized_df(df, vectorized_df):
    scaler, X_train_scaled, X_test_scaled, y_train, y_test, train_predictions, test_predictions = get_splits(df, vectorized_df)

    # drop all columns with an average of < 5% tfidf value
    X_train_reduced = X_train_scaled[X_train_scaled.columns[X_train_scaled.mean() > .05]]
    
    # make X_test have the same coluns as X_train
    reduced_list = X_train_reduced.columns.tolist()
    X_test_reduced = X_test_scaled[reduced_list]

    return X_train_reduced, X_test_reduced

def aggregate_columns(df):
    df['numbers'] = df['10'] + df['100'] + df['36']
    df['add+'] = df['add'] + df['additional']
    df['build+'] = df['build'] + df['building'] + df['built']
    df['contribute+'] = df['contribute'] + df['contributing'] + df['contributor']
    df['create+'] = df['create'] + df['created'] + df['creating']
    df['design+'] = df['design'] + df['designed']
    df['developed+'] = df['developed'] + df['developer'] + df['development']
    df['distributed+'] = df['distributed'] + df['distribution']
    df['follow+'] = df['follow'] + df['following']
    df['get+'] = df['get'] + df['getting']
    df['git+'] = df['git'] + df['github']
    df['install+'] = df['install'] + df['installation'] + df['installed'] + df['installing']
    df['provide+'] = df['provided'] + df['provides']
    df['recommend+'] = df['recommend'] + df['recommended']
    df['release+'] = df['release'] + df['released']
    df['require+'] = df['required'] + df['requirement'] + df['requires']
    df['run+'] = df['run'] + df['running']
    df['start+'] = df['start'] + df['started']
    df['support+'] = df['support'] + df['supported']
    df['use+'] = df['usage'] + df['use'] + df['used'] + df['useful'] + df['using']
    df['work+'] = df['work'] + df['working']

    replaced_columns = ['10', '100', '36', 'add', 'additional', 'build', 'building', 'built',
                        'contribute', 'contributing', 'contributor', 'create', 'created', 
                        'creating', 'design', 'designed', 'developed', 'developer', 'development', 
                        'distributed', 'distribution', 'follow', 'following', 'get', 'getting', 
                        'git', 'github', 'install', 'installation', 'installed', 'installing', 
                        'provided', 'provides', 'recommend', 'recommended', 'release', 'released',  
                        'required', 'requirement', 'requires', 'run', 'running', 'start', 'started', 
                        'support', 'supported', 'usage', 'use', 'used', 'useful', 'using', 'work', 'working']

    df = df.drop(labels=replaced_columns, axis=1)
    return df