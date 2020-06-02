import pandas as pd
import numpy as np

import os
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords



def basic_clean(df):
    '''
    This function takes in a df and a string for a column and
    returns the df with a new column named 'basic_clean' with the
    passed column text normalized.
    '''
    df['basic_clean'] = df['readme_contents'].str.lower()\
                    .replace(r'\<(.*?)\>','', regex = True)\
                    .replace(r'[^\w\s]', '', regex=True)\
                    .str.normalize('NFKC')\
                    .str.encode('ascii', 'ignore')\
                    .str.decode('utf-8', 'ignore')
    return df


def tokenize(df, col):
    '''
    This function takes in a df and a string for a column and
    returns a df with a new column named 'clean_tokes' with the
    passed column text tokenized and in a list.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    df['clean_tokes'] = df[col].apply(tokenizer.tokenize)
    return df


def stem(df, col):
    '''
    This function takes in a df and a string for a column name and
    returns a df with a new column named 'stemmed'.
    '''
    # Create porter stemmer
    ps = nltk.porter.PorterStemmer()
    
    # Stem each token from our clean_tokes Series of lists
    stems = df[col].apply(lambda row: [ps.stem(word) for word in row])
    
    # Join our cleaned, stemmed lists of words back into sentences
    df['stemmed'] = stems.str.join(' ')
    
    return df


def lemmatize(df, col):
    '''
    This function takes in a df and a string for column name and
    returns the original df with a new column called 'lemmatized'.
    '''
    # Create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Lemmatize each token from our clean_tokes Series of lists
    lemmas = df[col].apply(lambda row: [wnl.lemmatize(word) for word in row])
    
    # Join the cleaned and lemmatized tokens back into sentences
    df['lemmatized'] = lemmas.str.join(' ')
    df['lemmatized']= df['lemmatized'].replace(r'&#9','', regex = True)
    df['lemmatized']= df['lemmatized'].replace(r';','', regex = True)
    return df


def remove_stopwords(df, col):
    '''
    This function takes in a df and a string for column name and 
    returns the df with a new column named 'clean' with stopwords removed.
    '''
    # Create stopword_list
    stopword_list = stopwords.words('english')
    
    # Split words in column
    words = df[col].str.split()
    
    # Check each word in each row of the column against stopword_list and return only those that are not in list
    filtered_words = words.apply(lambda row: [word for word in row if word not in stopword_list])
    
    # Create new column of words that have stopwords removed
    df['clean_' + col] = filtered_words.str.join(' ')
    
    return df


def prep_readme_data(df):
    '''
    This function takes in the news articles df and
    returns the df with original columns plus cleaned
    and lemmatized content without stopwords.
    '''
    # Do basic clean on readme content
    df = basic_clean(df)
    
    # Tokenize clean readme content
    df = tokenize(df, 'basic_clean')
    
    # Lemmatize cleaned and tokenized readme content
    df = lemmatize(df, 'clean_tokes')
    
    # Remove stopwords from Lemmatized readme content
    df = remove_stopwords(df, 'lemmatized')

    # Count the number of links in each readme
    df['link_counts'] = df.readme_contents.str.count(r'https*')
    
    return df
