import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json
import math
import datetime
import collections
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re  # for preprocessing
from time import time
from collections import defaultdict
import spacy
import en_core_web_sm  # "small" model, trained using webpages
from matplotlib import cm


def amazon_reviews_to_dict(json_file_path):
    '''
    Creates a dictionary of dictionaries.
    Each entry of the datafile corresponds to a new entry in the dictionary.
    The entries in the main dictionary are labeled with an index from 0 to n-1 
    (where n is the total number of entries in the datafile).
    Each index value of the main dictionary contains a dictionary containing 
    the fields shown in the function below.
    '''
    reviews_dict = dict()  # initialize empty dictionary
    idx = 0

    with open (json_file_path, 'r') as file:
        for line in file:
            try:
                # Try to parse the current line
                entry = json.loads(line)

                # Extracts fields  
                review = {
                    'overall': entry.get('overall'),
                    'helpful': entry.get('helpful', [0, 0]),  # default value [0,0] if none
                    'productID': entry.get('asin'),
                    'reviewerID': entry.get('reviewerID'),
                    'unixReviewTime': entry.get('unixReviewTime'),
                    'reviewDate': entry.get('reviewTime'),
                    'reviewerName': entry.get('reviewerName'),
                    'summary': entry.get('summary'),
                    'reviewText': entry.get('reviewText')
                    }
                reviews_dict.update({idx: review})
                idx += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding json on line {idx}: {e}")
    return reviews_dict
      

def unixTime_to_humanTime(unixTime):
    '''
    Converts between unix time (number of seconds since Jan 1, 1970)
    and human time (in the format YYYY MM DD).
    '''
    return datetime.datetime.fromtimestamp(unixTime).strftime('%Y %m %d')


def humanTime_to_unixTime(humanTime):
    '''
    Converts between human time (in the format MM DD, YYYY)
    and unix time (number of seconds since Jan 1, 1970).
    '''
    # Parse the date from format mm dd, yyyy
    dt = datetime.datetime.strptime(humanTime, '%m %d, %Y')
    # Convert to Unix time
    return int(dt.timestamp())


def summary_statistics(data):
    '''
    Given a set of data, such as a list, the function calculates:
    - the mean (arithmetic average)
    - min, q1, med, q3, max values
    '''
    mean_val = np.mean(data)
    min_val = np.percentile(data, 0)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)
    max_val = np.percentile(data,100)

    return {
        "mean": mean_val,
        "min": min_val,
        "Q1": q1,
        "median": q2,
        "Q3": q3,
        "max": max_val
    }

def label_negative_sentiment(overallSentiment):
    if overallSentiment < 3:
        return 1
    else:
        return 0
    

def cleaning(review_spacy):
    
    # Lemmatizing and removing stopwords 
    # (review_spacy must be spaCy object)
    txt = [token.lemma_ for token in doc if not token.is_stop]
    #txt = [token.pos_ for token in doc]

    # Word2Vec uses the context words to learn to represent the vector for a word.
    # If a sentence only as one or two words, the benefit will be small.
    return ' '.join(txt)

