
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There aren't any necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.  
I used the following libraries and versions:

- [chardet](https://chardet.readthedocs.io/en/latest/) -- chardet==3.0.4
- [Flask](https://flask.palletsprojects.com/) -- Flask==1.1.2
- [joblib](https://joblib.readthedocs.io/) -- joblib==0.14.1
- [json](https://docs.python.org/3/library/json.html) -- json=2.0.9
- [nltk](https://www.nltk.org/) -- nltk==3.5
- [numpy](https://numpy.org/) -- numpy==1.18.3
- [pandas](https://pandas.pydata.org/) -- pandas==1.0.3
- [plotly](https://plotly.com/python/) -- plotly==4.6.0
- [requests](https://requests.readthedocs.io/en/master/) -- requests==2.23.0
- [scikit-learn](https://scikit-learn.org/) -- scikit-learn==0.22.2.post1
- [SQLAlchemy](https://www.sqlalchemy.org/) -- SQLAlchemy==1.3.16

## Project Motivation<a name="motivation"></a>

The aim of this project was to build a web application that would categorize disaster-related messages.  
This was done in two steps, with two pipelines:  
1. [Extract-Transform-Load pipeline](#etl), where I worked with two CSV files containing messages and categories, and end up with a database containing categorized messages.
2. [Machine Learning pipeline](#ml), where I built, tested and compared two different models to end up with one that predicts with a 95% accuracy new messages categories.


## File Descriptions <a name="files"></a>

- README.md, this file you're reading

#### Essentials:
- **web_app/** - contains the boilerplate code necessary to visualize the web application
- **web_app/README.md** - contains the necessary steps to run the web application in a local environment

#### Essentials not included:
- DisasterResponse.db (~6.5 MB) - it gets stored in your local environment after running *web_app/data/process_data.py*
- classifier.pkl (~997 MB) - it gets stored in your local environment after running *web_app/models/train_classifier.py*.

#### Pipeline Building Processes:
- <a name="etl">ETL Pipeline Preparation.ipynb</a>
- <a name="ml">ML Pipeline Preparation.ipynb</a>
- helpers.py

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
>The dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.  
– [Multilingual Disaster Response Messages, Appen Open Source Dataset](https://appen.com/datasets/combined-disaster-response-data/)