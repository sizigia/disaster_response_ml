
# import packages
import sys
import pandas as pd
import chardet
from sqlalchemy import create_engine


def load_data():
    # get encodings
    with open('messages.csv', "rb") as file:
        encoding_messages = chardet.detect(file.read())['encoding']
        
    with open('categories.csv', "rb") as file:
        encoding_categories = chardet.detect(file.read())['encoding']
        
    
    # read in file
    messages = pd.read_csv('messages.csv', encoding=encoding_messages)
    categories = pd.read_csv('categories.csv', encoding=encoding_categories)

    
    # clean data
    df = messages.merge(categories, on='id')
    
    categories = categories['categories'].str.split(";", expand=True)
    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda str_: str_[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda str_: str_[-1])
        categories[column] = categories[column].astype(str)
    
    df.drop(columns='categories', inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.dropna(subset=df.columns[4:], how='any', inplace=True)
    
    df.drop_duplicates(keep='first', inplace=True)

    
    # load to database
    engine = create_engine('sqlite:///etl.db')
    df.to_sql('etl.db', engine, index=False, if_exists='replace')


    # define features and label arrays
    X = df[df.columns[4:]].values
    y = df['message'].values

    return X, y
