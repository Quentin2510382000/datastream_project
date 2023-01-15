#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:52:23 2023

@author: quentin
"""

import re
from pathlib import Path

import nltk
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

def text_preprocessing(string, language):
    # remove accent
    string = unidecode.unidecode(string)

    # lowercase
    string = string.lower()

    # remove extra newlines
    string = re.sub(r"[\r|\n|\r\n]+", " ", string)

    # remove @tag
    string = re.sub(r"@[\S]+", "", string)

    # remove URL
    string = re.sub("https?://[\S]+", "", string)

    # remove hashtag and numbers
    string = re.sub("[^a-zA-Z]", " ", string)

    # tokenization
    string = word_tokenize(string)

    # remove stop words
    string = [word for word in string if word not in stopwords.words(language)]

    string = " ".join(word for word in string)

    return string