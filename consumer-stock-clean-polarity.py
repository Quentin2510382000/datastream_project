#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:30:26 2023

@author: quentin
"""

from kafka import KafkaConsumer,KafkaProducer
import json
from sentiment_analysis import text_preprocessing
from textblob import TextBlob

#

producer = KafkaProducer(bootstrap_servers='localhost:9092')
tpcs=['netflix','facebook','amazon','google','tesla']


consumer = KafkaConsumer()
consumer.subscribe(topics=tpcs)


#In this script we listen to the different kafka topics that we made
#We subscribe to the 5 topics we created, and we pre-process all the
# tweets before associating a polarity that represents the sentiment with Text Blob library. 
   
while True:
  
    raw_messages = consumer.poll(
        timeout_ms=100, max_records=200
    )
    
    for topic_partition, msg in raw_messages.items():
        
        tweet=json.loads(msg[0].value)
        print(tweet)
        if tweet["lang"] == "fr":
            language = "french"
        elif tweet["lang"] == "en":
            language = "english"
        else:
            continue
        string=tweet["tweet"]
        string_preprocessed = text_preprocessing(string, language)
        polarity = TextBlob(string_preprocessed).sentiment.polarity
        dictionnary={'tweet':tweet['tweet'],'lang':tweet['lang'],'company':tweet['company'],'polarity':polarity}
        producer.send(tweet['company']+'_polarity', json.dumps(dictionnary).encode('utf-8'))
        
        

    

        
        

