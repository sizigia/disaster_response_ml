import sys
import nltk
#nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import pickle
import re

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table(database_filepath, engine)
    
    X = df['message']
    y = df[df.columns[-36:]]
    
    return X, y, df.columns[-36:]


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, 
                                                     learning_rate=0.5)))
    ])
    
    parameters = {
    'clf__estimator__learning_rate': [0.1, 0.2, 0.5],
    'clf__estimator__n_estimators': [100, 200, 300]
    }

    cv = GridSearchCV(model, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for idx, col in enumerate(category_names):
        set_y_pair = (y_test[col], y_pred[:, idx])
        avg='weighted'
        rep_col = "{}\n\tPrecision: {:.2f}%\n\tRecall: {:.2f}%\n\tF1 Score: {:.2f}%\n\tAccuracy: {:.2f}%".format(col,
                                                                                 precision_score(*set_y_pair, average=avg), 
                                                                                 recall_score(*set_y_pair, average=avg), 
                                                                                 f1_score(*set_y_pair, average=avg))
    print(rep_col)
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
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