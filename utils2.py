# LIBRARIES/MODULES USED IN THIS PREPROCESSING FILE

# Standard libraries
import json
import datetime
from typing import Any, Dict

# Data handling
import numpy as np
import pandas as pd

# NLP
import spacy
from nltk.corpus import stopwords
from spacy.tokens import Token

# Load spaCy with only tokenizer and POS tagger (disable parser and NER)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Create stopword set, keeping sentiment/contrast words
base_stopwords = set(stopwords.words('english'))
keep_words = {
    'no', 'not', 'nor', 'but','less','least', 'against',
    'very','too', 'like', 'more','most','only','just','same','again'
}
stop_words = base_stopwords - keep_words  # Remove from stopwords so they are retained

def preprocess_data(json_filepath, max_tokens=256):
    """
    Full preprocessing pipeline:
    1. Load JSON into dictionary by calling reviews_to_dict()
    2. Convert to DataFrame
    3. Preprocess text:
       Tokenize (removing punctuation and extra spaces)
       Lemmatize
       Lowercase
       Remove stopwords (using modified NLTK list)
    4. Store length of review (in number of preprocessed tokens)    
    5. Truncate long reviews to max_tokens (=256 as default), 
        keeping the first and last (and removing the ones in between)
    6. Reconstruct processed text separating tokens with a single space
    7. Inform on process evolution
    8. Add reviewLength and processedText to df
    Returns:
        df: pandas DataFrame with columns:
            'label', 'helpful', 'age', 'reviewLength', 'processedText'
    """
    
    # 1. Load JSON into dictionary
    print("Sending request to create dictionary.")
    reviews_dict = reviews_to_dict(json_filepath)
    print("Dictionary created and loaded.")

    # 2. Convert to DataFrame
    df = pd.DataFrame.from_dict(reviews_dict, orient='index')
    print("Dataframe created and loaded.")

    # 3. Preprocess text
    processed_texts = [] # Initialize list for storing all preprocessed reviews
    review_lengths = []  # Initialize list for preprocessed lengths (prior to truncation)
    count = 0            # For verbose purposes

    for doc in nlp.pipe(df['text'], batch_size=1000):

        tokens = []  # Initialize list for storing preprocessed tokens of current review
        
        tokens = [
            # Lemmatize and convert to lowercase
            token.lemma_.lower() for token in doc
            # Remove punctuation, extra spaces and stopwords
            if not token.is_punct 
            and not token.is_space 
            and token.lemma_.lower() not in stop_words
        ]
        
        # 4. Store length of review (in tokens) before truncation
        review_lengths.append(len(tokens))

        # 5. Truncate if needed
        if len(tokens) > max_tokens:
            tokens = tokens[:(max_tokens // 2)] + tokens[-(max_tokens // 2):]

        # 6. Reconstruct processed text
        processed_texts.append(" ".join(tokens))

        # 7. Inform on process evolution
        if count % 5000 == 0:
            print (f"{count} reviews preprocessed")
        count +=1

    print("Preprocessing finished. Adding preprocessed data to dataframe.")
    
    # 8. Add to dataframe
    df["reviewLength"] = review_lengths
    print("Review length (in tokens) added to dataframe")
    
    df["processedText"] = processed_texts
    print("Processed text added to dataframe")

    return df

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


def reviews_to_dict(json_filepath: Any) -> Dict[int, dict]:
    '''
    Opens json filepath given and reads each line. 
    For each line, creates a 'review' dictionary with fields: 
        'label': 1 if negative sentiment (less than 3 stars), 0 otherwise
        'helpful': same list of two integers as in 'helpful'
        'age': review age in days since 07 24, 2014 (all reviews will be at least 1 day old)
        'text': text of the review
    Each line is itself encoded as an integer entry of the main dictionary 
    (from entry 0 to n-1, where n is the total number of reviews)
    Returns the dictionary of dictionaries.
    '''
    reviews_dict = {}
    idx = 0
    with open(json_filepath, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # Binarize 'overall' score: <3 = 1 (negative), >=3 = 0 (positive)
                if entry.get('overall',3) < 3:
                    negSentiment = 1 
                else:
                    negSentiment = 0
                
                # Select review text (use 'summary' if longer than 'reviewText')
                reviewText = entry.get('reviewText', '').strip()
                summary = entry.get('summary', '').strip()
                if len(summary) > len(reviewText):
                    reviewText = summary
                
                # Select 'unixReviewTime' and convert to 'reviewAge'
                tmax = humanTime_to_unixTime('07 24, 2014')  # one day after most recent date
                unixReviewTime = entry.get('unixReviewTime')
                if unixReviewTime is None:
                    reviewAge = None
                else:
                    reviewAge = (tmax - unixReviewTime) // (60*60*24)  # calculate age in seconds, then divide by number of seconds in a day

                review = {
                    'label': negSentiment,
                    'helpful': entry.get('helpful', [0, 0]),
                    'age': reviewAge,
                    'text': reviewText
                }
                reviews_dict.update({idx: review})
                idx += 1
            except json.JSONDecodeError:
                continue
    print("Dictionary created.")
    return reviews_dict

