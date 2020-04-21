
import sys
import pandas as pd
import chardet
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # get encodings
    with open(messages_filepath, "rb") as file:
        encoding_messages = chardet.detect(file.read())['encoding']
        
    with open(categories_filepath, "rb") as file:
        encoding_categories = chardet.detect(file.read())['encoding']
        
    # read in file
    messages = pd.read_csv(messages_filepath, encoding=encoding_messages)
    categories = pd.read_csv(categories_filepath, encoding=encoding_categories)
    
    # merge dataframes
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    categories = df['categories'].str.split(";", expand=True)
    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda str_: str_[:-2])
    categories.columns = category_colnames
    
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
        categories[column] = categories[column].apply(lambda str_: str_[-1])
        categories[column] = categories[column].astype(int)
    
    df.drop(columns='categories', inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.dropna(subset=df.columns[4:], how='any', inplace=True)
    df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filename):
     # load to database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
