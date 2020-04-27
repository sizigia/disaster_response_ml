import json
import plotly
import pandas as pd

import nltk
nltk.download(['averaged_perceptron_tagger', 'wordnet'])
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
