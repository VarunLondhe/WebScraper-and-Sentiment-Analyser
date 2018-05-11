# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:19:26 2018

@author: Varun.Londhe
"""
import re
import os
import errno
from urllib import parse
import math
import statistics as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer
import Constants

# Read a text file to fetch the company name for scaping
def reuters_scrapper(url):
    """Returns the Date, Source, Heading and Article content from the URL passed."""
    all_contents = []
    try:
        web_page = requests.get(url)
        web_page.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        pass
    reuters_soup = BeautifulSoup(web_page.content, "lxml")
    all_p_tags = []

    for tag in reuters_soup.find_all(Constants.P):
        all_p_tags.append(tag)
    article = Constants.EMPYT_STRING

    for x in all_p_tags[:-2]:
        article = article + x.text
    source = parse.urlparse(url).hostname.split(".")[Constants.INDEX_1]
    all_contents.append(source)
    all_contents.append(reuters_soup.title.text.lstrip())
    all_contents.append(article)
    return all_contents
# *********************************************************************************************
FILE = open("C:/Users/varun.londhe/Documents/Python Practice/Reuteurs/Company Repository.txt", "r")
READCOMPANY = FILE.read().splitlines()
# headLink = Constants.REUTERS_HEAD_LINK
# tailLink = Constants.REUTERS_TAIL_LINK
READCOMPANY
FILE.close()
df = pd.DataFrame()
for company in READCOMPANY:
    searchUrl = Constants.REUTERS_HEAD_LINK + company + Constants.REUTERS_TAIL_LINK
    try:
        searchResponse = requests.get(searchUrl)
        searchResponse.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        pass
    searchSoup = BeautifulSoup(searchResponse.content, "lxml")
    findTotalResults = searchSoup.find(Constants.SPAN, \
                                       {Constants.CLASS: \
                                        "search-result-count search-result-count-num"}).text
    findCompany = searchSoup.find(Constants.SPAN, {Constants.CLASS: "search-keyword"}).text
# Iterations to calcualte the number of pages populated by the search operation
    interations = math.ceil(int(findTotalResults)/10)
    interations
    dicts = {}
    for i in range(Constants.INDEX_1, interations + Constants.INDEX_1):
        newSearchUrl = searchUrl + Constants.REUTERS_DYNAMIC_LINK + str(i)
        newSearchUrl
#        newSearchResponse = requests.get(newSearchUrl)
        try:
            newSearchResponse = requests.get(newSearchUrl)
            newSearchResponse.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
            pass
        newSearchSoup = BeautifulSoup(newSearchResponse.content, "lxml")
        findIndividualSearch = newSearchSoup.find_all(Constants.DIV, \
                                                      {Constants.CLASS: "search-result-content"})
        len(findIndividualSearch)
        for ind in findIndividualSearch:
        #    links.append(ind.find('a', {"href": re.compile("/")}).get('href'))
            date = ind.find(Constants.H5, \
                            {Constants.CLASS: "search-result-timestamp"}).text.replace(":", ".")
            date = date.replace('am', ' AM').replace('.', ':') if 'am' \
                    in  date else date.replace('pm', ' PM').replace('.', ':')
            dicts[date] = Constants.REUTERS_HEADER + ind.find(Constants.A,
                                                              {Constants.HREF: re.compile("/")}\
                                                              ).get(Constants.HREF)

    dicts
    labels = ["Company", "Date", "Source", "URL", "Heading", "Content"]
    for key, value in list(dicts.items()):
        alldetails = reuters_scrapper(value)
        extractedDate = key
        if alldetails and findCompany in alldetails[2]:
            alldetails.insert(0, findCompany)
            alldetails.insert(Constants.INDEX_1, key)
            alldetails.insert(3, value)
            df = pd.concat([df, pd.DataFrame([alldetails], columns=labels)])
        else:
            del dicts[key]
# *********************************************************************************************
def find_whole_word(w):
    """A REGEX to find out the location of metadata about the article"""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

# Calculate Sentiment Scores!

POLARITY_TEXTBLOB = []
SUBJECTIVITY = []
POLARITY_VADER = []
TEXTBLOB_FULL_ARTICLE = []
POLARITY_ARTICLE = []
for news in df["Content"]:
    VADER_ARTICLE_COMPOUND = []
    TEXTBLOB_ARTICLE_POLARITY = []
    TEXTBLOB_ARTICLE_SUBJECTIVITY = []
    try:
        a = find_whole_word('(Reuters)')(news).span()[1]
        b = find_whole_word('Reporting by')(news).span()[0]
        sentences = PunktSentenceTokenizer().tokenize(news[a + 4: b])
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
    TEXTBLOB_FULL_ARTICLE.append(TextBlob(news).sentiment.polarity)
    POLARITY_ARTICLE.append(SentimentIntensityAnalyzer().polarity_scores(news)["compound"])
FILENAME = "C:/Users/varun.londhe/Documents/Python Practice/Reuteurs/PDVSA/"
if not os.path.exists(os.path.dirname(FILENAME)):
    try:
        os.makedirs(os.path.dirname(FILENAME))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
df["Polarity TextBlob"] = pd.Series(POLARITY_TEXTBLOB, index=df.index)
df["Polarity TextBlob Full Article"] = pd.Series(TEXTBLOB_FULL_ARTICLE, index=df.index)
df["Subjectivity TextBlob"] = pd.Series(SUBJECTIVITY, index=df.index)
df["Polarity Vader"] = pd.Series(POLARITY_VADER, index=df.index)
df["Vader article"] = pd.Series(POLARITY_ARTICLE, index=df.index)
df = df.drop_duplicates(['URL'], keep='first')
df = df.drop_duplicates(['Content'], keep='last')
df.to_csv("C:/Users/varun.londhe/Documents/Python Practice/Reuteurs/ReutersTimeModified.csv", \
          encoding='utf-8-sig', index=False)
