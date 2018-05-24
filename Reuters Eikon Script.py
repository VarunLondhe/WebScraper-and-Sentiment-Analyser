# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:34:08 2018

@author: Varun.Londhe
"""
import os
import errno
import re
import statistics as st
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer
import eikon as ek


def getNewsText(storyId):
    """
    Fetches the news story from the story ID and returns the text from soup object
    """
    newsText = ek.get_news_story(storyId)
    if newsText:
        #create a BeautifulSoup object from our HTML news article
        soup = BeautifulSoup(newsText, "lxml")

    return soup.get_text()

#******************************************************************************

def find_whole_word(w):
    """A REGEX to find out the location of metadata about the article"""
    return re.compile(r'\b({0}\W+)'.format(w)).search

#******************************************************************************

def removeIFRMetaData(text):
    """
    Removes the head and tail part containing the News Source and reporter 
    details for IFR
    """
    star_index = 0
    end_index = len(text) if isinstance(text, str) else 0
    text = str(text) if type(text) is not str else text
    if end_index != 0:
        if find_whole_word('(IFR)')(text):
            star_index = find_whole_word('(IFR)')(text).span()[1]
        if find_whole_word('(Reporting by)')(text):
            end_index = find_whole_word('(Reporting by)')(text).span()[0]

    return text[star_index: end_index]

#******************************************************************************

def removeReutersMetaData(text):
    """
    Removes the head and tail part containing the News Source and reporter 
    details for Reuters
    """
    star_index = 0
    end_index = []
    end_index.append(len(text) if isinstance(text, str) else 0)
    text = str(text) if type(text) is not str else text
    if end_index[0] != 0:
        if re.compile(r'\(Reuters\) - ').search(text):
            star_index = re.compile(r'\(Reuters\) - ').search(text).span()[1]
        if re.compile(r'\WReporting by\W').search(text):
            end_index.append(re.compile(r'\WReporting by\W').search(text).span()[0])
        elif find_whole_word('(Additional reporting)')(text):
            end_index.append(find_whole_word('(Additional reporting)')(text).span()[0])
        elif find_whole_word('(Editing by)')(text):
            end_index.append(find_whole_word('(Editing by)')(text).span()[0])
        elif find_whole_word('(Compiled by)')(text):
            end_index.append(find_whole_word('(Compiled by)')(text).span()[0])

    return text[star_index: min(end_index)]

#******************************************************************************

def removePlattsMetaData(text):
    """
    Removes the head and tail part containing the News Source and reporter 
    details for Platts
    """
    end_index = []
    star_index = [0]
    end_index.append(len(text) if isinstance(text, str) else 0)
    text = str(text) if isinstance(text, str) else text
    if end_index[0] != 0:
        if re.compile(r'\bGMT\W+\w+ GMT\W').search(text):
            star_index.append(re.compile(r'\bGMT\W+\w+ GMT\W').search(text).span()[1])
        if re.compile(r'\bGMT\w+').search(text):
            star_index.append(re.compile(r'\bGMT\w+').search(text).span()[1])
        if re.compile(r'\W+PLEASE SEND\b').search(text):
            end_index.append(re.compile(r'\W+PLEASE SEND\b').search(text).span()[0])
        elif re.compile(r'\W+Platts Global\b').search(text):
            end_index.append(re.compile(r'\W+Platts Global\b').search(text).span()[0])
        elif re.compile(r'\bSource\W').search(text):
            end_index.append(re.compile(r'\bSource\W').search(text).span()[0])

    return text[max(star_index): min(end_index)]

#******************************************************************************

def removeENPNWSMetaData(text):
    """
    Removes the head and tail part containing the News Source and reporter 
    details for ENPNWS
    """
    star_index = 0
    end_index = len(text) if isinstance(text, str) else 0
    
    if re.compile(r'\bENPUBLISHINGRelease\W+\w+\W+\w+\W+\w+\W+').search(text):
        star_index = re.compile(r'\bENPUBLISHINGRelease\W+\w+\W+\w+\W+\w+\W+').search(text).span()[1]
    
    if re.compile(r'Contact\WTel\W').search(text):
        end_index = re.compile(r'Contact\WTel\W').search(text).span()[0]
    
    return text[star_index:end_index]

#******************************************************************************

def removePRNewswireMetaData(text):
    """
    Removes the head and tail part containing the News Source and reporter 
    details for PRNewswire
    """
    star_index = 0
    end_index = len(text) if isinstance(text, str) else 0
    
    if re.compile(r'\W+Notes\W+\w+\W+\w+\W+').search(text):
        star_index = re.compile(r'\W+Notes\W+\w+\W+\w+\W+').search(text).span()[1]
    
    if re.compile(r'\w+Copyright\W+\w+\W+').search(text):
        end_index = re.compile(r'\w+Copyright\W+\w+\W+').search(text).span()[0]
    
    return text[star_index:end_index]

#******************************************************************************

def getSentiment(text):
    """
    Find out the sentiment for the given article
    """
    sentences = PunktSentenceTokenizer().tokenize(text)
    polarity_score_list = []

    for sentence in sentences:
        polarity_score_list.append(SentimentIntensityAnalyzer().polarity_scores(sentence)\
                                   ["compound"])
    try:
        return st.mean(polarity_score_list)
    except st.StatisticsError:
        return 0

#******************************************************************************

def updateSentimentDataFrame(df):
    """
    Fetch the news story and append polarity scores in the dataframe
    """

    df['News text'] = df['storyId'].apply(lambda x: getNewsText(x))
#    df = cleanNewsText(df)
    df["News text"] = df["News text"].apply(lambda x: getCleanText(x))
#    df["News Text"] = df[["News text", "sourceCode"]].apply(lambda x: \
#                                                            findMetaDataFunction(x[0], x[1]), \
#                                                            axis=1)
    df["News text"] = df[["News text", "sourceCode"]].apply(lambda x: \
                                                            findMetaDataFunction(x[0], x[1]), \
                                                            axis=1)
    df["News text"].replace('', pd.np.nan, inplace=True)
    df=df.dropna()
    df['Polarity Score'] = df['News text'].apply(lambda x: getSentiment(x))
    return df

#******************************************************************************

def establishConnectionWithReuters(app_id):
    """
    Establish the connection wth EIKON Reuters
    """
    ek.set_app_id(app_id)

    return ek

#******************************************************************************

def fetchNewsHeadlines(ek, companyName):
    """
    Mine the headlines for the given company
    """
#    companyName = 'R:RDSa.L'
    df = ek.get_news_headlines(companyName+' AND Language:LEN', date_to=\
                           "2018-05-22", count=100)

    return df

#******************************************************************************

def getCleanText(x):
    """
    Combines the multiple lined string into one liner
    """
    return "".join(x.splitlines())

#******************************************************************************

def findMetaDataFunction(x, y):
#    print("the x is ", x[1:10], "and y is ", y)
    if y == "NS:IFR":
        x = removeIFRMetaData(x)
        return x
    
    elif y == "NS:RTRS":
        x = removeReutersMetaData(x)
        return x
    
    elif y == "NS:PLTS":
        x = removePlattsMetaData(x)
        return x
    
    elif y == "NS:ENPNWS":
        x = removeENPNWSMetaData(x)
        return x
    
    elif y == "NS:PRN":
        x = removePRNewswireMetaData(x)
        return x
    
    return x

#******************************************************************************
# app_id = "<Your App ID>"
ek = establishConnectionWithReuters(app_id)
#R:RDSa.L AND Language:LEN
#R:CW158243342= AND Language:LEN
#R:VE059352415= AND Language:LEN
#P:[Gulf Petrochem FZC] AND Language:LEN
df_row_merged = pd.DataFrame()
company_list = ['R:RDSa.L', 'R:CW158243342=', 'R:VE059352415=', 'P:[Gulf Petrochem FZC]']
for company in company_list:
    ek = establishConnectionWithReuters(app_id)
    company = 'P:[Gulf Petrochem FZC]'
    print("Starting for ", company)
    testdf = fetchNewsHeadlines(ek, company)
    #companyName = 'R:RDSa.L'
    testdf = updateSentimentDataFrame(testdf)

    testdf = testdf.drop_duplicates(['News text'], keep='first')
#    testdf["Company"] =
    df_row_merged = pd.concat([testdf, df_row_merged], ignore_index=True)

#******************************************************************************
# Write the data dump into a CSV file

FILENAME = "C:/Users/varun.londhe/Documents/Python Practice/Reuteurs/"
if not os.path.exists(os.path.dirname(FILENAME)):
    try:
        os.makedirs(os.path.dirname(FILENAME))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
#Uncomment this!!!!!!!!!!!!!!!!!!
#df_row_merged.to_csv(FILENAME + "AllReuters.csv", encoding='utf-8-sig', index=False)

testdf = pd.read_csv('C:/Users/varun.londhe/Documents/Work/Observations for' \
                     + 'Sentiment Analysis/EIKON Connection/AllReuters.csv')

#******************************************************************************
