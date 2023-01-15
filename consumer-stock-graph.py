#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:44:20 2023

@author: quentin
"""

from kafka import KafkaConsumer,KafkaProducer
import json
import pandas as pd
import csv


tpcs = ['netflix_polarity','facebook_polarity','amazon_polarity','google_polarity','tesla_polarity']

consumer = KafkaConsumer()
consumer.subscribe(topics=tpcs)

#This script aims to create for each company a csv file
# where we will find the sentiment analysis of each tweet associated with the company.
#in  order to archive the sentiment analysis and permits the financial dashboard to get these datas

filename = "tweet_sentiment"
for i in tpcs:
    f = open(filename+'_'+i+'.csv', 'w')
    writer_object = csv.writer(f)
    
    writer_object.writerow([i,"polarity"])
    f.close()



while True:
    
    raw_messages = consumer.poll(
        timeout_ms=100, max_records=200
    )
    
    for topic_partition, msg in raw_messages.items():
        
        
     
        with open(filename+'_'+(topic_partition.topic)+'.csv', 'a') as f:
            
        
            writer_object = csv.writer(f)
         
      
            dictionnary=json.loads(msg[0].value)
            l=[dictionnary["tweet"],dictionnary["polarity"]]
            
            writer_object.writerow(l)
 
    
    
    