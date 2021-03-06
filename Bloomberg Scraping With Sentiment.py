# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:57:08 2018

@author: Varun.Londhe
"""
import os
import errno
import re
from urllib import parse
import statistics as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer
import Constants

def bloomberg_scrapper(url):
    """Returns the Date, Source, Heading and Article content from the URL passed."""
    all_contents = []
    try:
        web_page = requests.get(url)
        web_page.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        pass
    bloomberg_soup = BeautifulSoup(web_page.content, "lxml")
    try:
        extract_date = bloomberg_soup.find(Constants.NO_SCRIPT).text.replace(":", ".")
        test = bloomberg_soup.find(Constants.DIV, \
                                   {Constants.CLASS: "transporter-item current"}).find_all("p")
        article = Constants.EMPYT_STRING
        for content in test:
            article = article + content.text
        source = parse.urlparse(url).hostname.split(".")[Constants.INDEX_1]
        all_contents.append(extract_date)
        all_contents.append(source)
        all_contents.append(bloomberg_soup.title.text)
        all_contents.append(article)
    except AttributeError as err:
        pass
    return all_contents
# *************************************************************************************************

FILE = open("C:/Users/varun.londhe/Documents/Python Practice/Reuteurs/Company Repository.txt", "r")
READCOMPANY = FILE.read().splitlines()
READCOMPANY
FILE.close()
df = pd.DataFrame()
LABELS = ["Company", "Date", "Source", "URL", "Heading", "Content"]
for company in READCOMPANY:
    COUNTER = 1
    while COUNTER != None:
        searchUrl = Constants.BLOOMBERG_HEADLINK + company + Constants.BLOOMBERG_TAILLINK \
        + str(COUNTER)
        try:
            searchResponse = requests.get(searchUrl)
            searchResponse.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
            pass
        bloomberg_search_soup = BeautifulSoup(searchResponse.content, "lxml")
        findCompany = bloomberg_search_soup.find(Constants.INPUT, \
                                                 {Constants.CLASS: "settings-search-box__input"})\
                                                 .get('value')
        try:
            findNext = bloomberg_search_soup.find(Constants.A, \
                                                  {Constants.CLASS: "content-next-link"}).text
            COUNTER += 1
        except AttributeError:
            COUNTER = None
            print("The counter is now none for ", findCompany)

        findIndividualSearch = bloomberg_search_soup.find_all(Constants.H1, \
                                                              {Constants.CLASS:\
                                                               "search-result-story__headline"})
        for ind in findIndividualSearch:
            article_url = ind.find(Constants.A, {Constants.HREF: re.compile("/")}).\
            get(Constants.HREF)
            alldetails = bloomberg_scrapper(article_url)
            if alldetails:
                alldetails.insert(0, findCompany)
                alldetails.insert(3, article_url)
                df = pd.concat([df, pd.DataFrame([alldetails], columns=LABELS)])
            else:
                continue

# *************************************************************************************************
def find_whole_word(w):
    """A REGEX to find out the location of metadata about the article"""
    return re.compile(r'({0})'.format(w), flags=re.IGNORECASE).search

POLARITY_TEXTBLOB = []
SUBJECTIVITY = []
POLARITY_VADER = []
VADER_ARTICLE_COMPOUND = []
TEXTBLOB_ARTICLE_POLARITY = []
TEXTBLOB_ARTICLE_SUBJECTIVITY = []
for news in df["Content"]:
    try:
        a = find_whole_word('/Bloomberg')(news).span()[1]
#       b = find_whole_word('Reporting by')(news).span()[0]
        sentences = PunktSentenceTokenizer().tokenize(news[a + 1: ])
    except:
        sentences = PunktSentenceTokenizer().tokenize(news)

    for sentence in sentences:
        vaderAnalyzer = SentimentIntensityAnalyzer()
        vs = vaderAnalyzer.polarity_scores(sentence)
        textBlobAnalyzer = TextBlob(sentence)
        VADER_ARTICLE_COMPOUND.append(vs["compound"])
        TEXTBLOB_ARTICLE_POLARITY.append(textBlobAnalyzer.sentiment.polarity)
        TEXTBLOB_ARTICLE_SUBJECTIVITY.append(textBlobAnalyzer.sentiment.subjectivity)
    POLARITY_TEXTBLOB.append(st.mean(TEXTBLOB_ARTICLE_POLARITY))
    SUBJECTIVITY.append(st.mean(TEXTBLOB_ARTICLE_SUBJECTIVITY))
    POLARITY_VADER.append(st.mean(VADER_ARTICLE_COMPOUND))

df["Polarity TextBlob"] = pd.Series(POLARITY_TEXTBLOB, index=df.index)
df["Subjectivity TextBlob"] = pd.Series(SUBJECTIVITY, index=df.index)
df["Polarity Vader"] = pd.Series(POLARITY_VADER, index=df.index)
FILENAME = "C:/Users/varun.londhe/Documents/Python Practice/Bloomberg/"

if not os.path.exists(os.path.dirname(FILENAME)):
    try:
        os.makedirs(os.path.dirname(FILENAME))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
df = df.drop_duplicates(['URL'], keep='first')
df.to_csv("C:/Users/varun.londhe/Documents/Python Practice/Bloomberg/Bloomberg_test_1M_Mean.csv", \
          encoding='utf-8', index=False)
