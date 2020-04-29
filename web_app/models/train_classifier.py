import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['averaged_perceptron_tagger', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Creates an Engine instance with the path provided, and reads the SQL table that stores the cleaned data.
    Returns features, labels and category names for the dataset.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('DisasterResponseData', engine)
    
    X = df['message']
    y = df[df.columns[-36:]]
    
    return X, y, df.columns[-36:]


def tokenize(text):
    """
    Transforms a text to clean tokens, where every token is a word converted to lower case,
    passed to a part-of-speech tagger and lemmatized accordingly.
    Words recognized as stopwords are ommitted.
    
    Input:
        text (str)
        
    Output:
        clean_tokens (list): list of clean tokens (words converted to lower case and lemmatized)
        
    """
    
    tokenizer = RegexpTokenizer('\w+')
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(text.lower())
    
    clean_tokens = []
    
    for word, tag in pos_tag(tokens):
        if tag[0] in ['A', 'R', 'N', 'V']:
            tag = tag[0].lower()
            clean_token = lemmatizer.lemmatize(word, pos=tag)
        else:
            clean_token = word
            
        if clean_token not in stopwords.words('english'):
            clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """
    No input needed. Returns a pipeline with the next steps:
        1. vect - Converts a collection of text documents to a matrix of token counts
        2. tfidf - Transforms a count matrix to a normalized *term-frequency* or 
                  *term-frequency times inverse document-frequency* representation
        3. clf - Multi target random forest classification. It reuses the solution 
                    of the previous call to fit and add more estimators to the ensemble.    
    """
    
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(warm_start=True))),
    ], verbose=True)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes the model, X and Y test set and category names, and returns precision, recall and F1 score 
    for every feature in the dataset, and the overall accuracy of the model.
    
    Input:
        model ():
        X_test (pandas.core.series.Series): a subset of Y with the purpose of testing the model
        Y_test (pandas.core.series.Series): predictions made with X_test by the model
        category_names (): 
        
    Output:
        Prints out the following format
            feature_name
                Precision: __%
                Recall: __%
                F1 Score: __%
                
                ...
                
                Accuracy Score: __%
                
        And also returns the full value of accuracy.
    """
    
    Y_pred = model.predict(X_test)
    
    for idx, col in enumerate(category_names):
        set_Y_pair = (Y_test[col], Y_pred[:, idx])
        avg='weighted'
        rep_col = "{}\n\tPrecision: {:.2f}%\n\tRecall: {:.2f}%\n\tF1 Score: {:.2f}%\n".format(col,
                                                                                 precision_score(*set_Y_pair, average=avg), 
                                                                                 recall_score(*set_Y_pair, average=avg), 
                                                                                 f1_score(*set_Y_pair, average=avg))
        print(rep_col)
        
    print('Accuracy Score: {:.2f}%'.format(np.mean(Y_test.values == Y_pred)))

    return np.mean(Y_test.values == Y_pred)


def save_model(model, model_filepath):
    """
    Takes in the trained model and a path where to store it.
    Dumps the model in a pickle to reuse it later.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
