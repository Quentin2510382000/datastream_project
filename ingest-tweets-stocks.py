#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:02:29 2023

@author: quentin
"""


import tweepy
import json
from kafka import KafkaConsumer, KafkaProducer
consumer_key='mCJpDwDhni9eP0qkozfV6ihIq'
consumer_secret='nh9mOkXK34f9yPPLksHSFrXm83wIPCUq15ttlUk4vU8X4x1ZSr'
access_token='1271435978008408068-pbqBJvz4TVLMLMF3GOYS3AKHjtVhqN'
access_token_secret='e41FvGyic7EeFNYJEzPOMVM7zsddtwQeD2qLbLtiJcLsu'

import tweepy
import json
from kafka import KafkaConsumer, KafkaProducer


#This code permit us to get all the tweets talking about the 5 stocks that we have selected
#It should be noted that for this analysis, we have chosen different actions from those we have analyzed in the notebooks
#we provide you with. This is because we needed a famous company to find enough tweets on Twitter.
#we also use kafka in order to structure our stream pipeline.


client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAANEBeAEAAAAAvwaVl0s%2BanXo1S%2BJYVRcu2baYfU%3D9HtlIJoMBZcE6Hd6QF5wloJ75SW7562rg7gkkfL98PTF0hNJnn')

keywords = ['netflix','facebook','amazon','google','tesla']

query = ' OR '.join(keywords)

producer = KafkaProducer(bootstrap_servers='localhost:9092')

#we have define a producer that permit us to send in 5 differnet kafka topics, all the tweets talking about the 5 company.

for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
tweet_fields=['lang'], max_results=100).flatten(limit=10000):
    
    for keyword in keywords:
        
        if keyword in (tweet["text"]).lower():
            dictionnary={'tweet':tweet['text'],'lang':tweet['lang'],'company':keyword}
            producer.send(keyword, json.dumps(dictionnary).encode('utf-8'))
    



            


