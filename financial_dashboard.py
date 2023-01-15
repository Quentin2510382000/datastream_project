#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sat Jan 14 17:12:15 2023

@author: quentin
"""

import streamlit as st
from PIL import Image
import pyodbc
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
import plotly.graph_objs as go
import yfinance as yf
from plotly.subplots import make_subplots
import numpy as np
from river import metrics
from river import utils
from river.stream import iter_pandas
from river import linear_model 
from river import preprocessing
from river import linear_model
from river import compose
from river import optim 
from river.tree import HoeffdingTreeClassifier
from river.neighbors import KNNClassifier
from river.ensemble import AdaBoostClassifier, AdaptiveRandomForestClassifier
from river import feature_extraction
from river import stats
from river import evaluate
import numpy as np
import pandas as pd
import time
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime


def getSentiment(score):
  if score < 0:
    return 'Negative'
  elif score ==0:
    return 'Neutral'
  else:
    return 'Positive'




  


def make_model(alpha):
    scale = preprocessing.StandardScaler()

    learn = linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(0.03),
        loss=optim.losses.Quantile(alpha=alpha)
    )

    model = scale | learn
    model = preprocessing.TargetStandardScaler(regressor=model)

    return model

models = {
    '0.1': make_model(alpha=0.1),
    '0.3': make_model(alpha=0.3),
    'center': make_model(alpha=0.5),
    '0.6': make_model(alpha=0.6),
    '0.8': make_model(alpha=0.8),
    '0.95': make_model(alpha=0.95)
}
    




def Main():
    
    #This code have been built to present the prediction of stream financial data. 
    # Since the U.S. market will not be open during our presentation time, we have obtained the financial data for the 5 stocks for 13/01/2022 and we have feed our algo as these were a stream
    #We have built our financial dashboard in order to find the real time price of each stocks, with the prediction of our model based on knn. You will also find the result of the use of the 
    #twitter API (number of neutral, positive and negative tweets with the mean of the sentiment for each stock in real time)
    #you can run this script in order to see the financial dashboard app locally with the following command: streamlit run financial_dashboard.py (write it in a terminal)
    
    list_company=["META","NETFLIX","AMAZON","GOOGLE","TESLA"]
    
    with st.sidebar:
        
        st.warning("DASHBOARD")
        price=0
        change=0
        m1=st.empty()
        m2=st.empty()
        m3=st.empty()
        m4=st.empty()
        m5=st.empty()
        #display the price and return of each stocks in real time
        m1.metric('META', f'{round(price,1):,}', f'{round(change*100,2)}%')
        m2.metric('NETFLIX', f'{round(price,1):,}', f'{round(change*100,2)}%')
        m3.metric('AMAZON', f'{round(price,1):,}', f'{round(change*100,2)}%')
        m4.metric('GOOGLE', f'{round(price,1):,}', f'{round(change*100,2)}%')
        m5.metric('TESLA', f'{round(price,1):,}', f'{round(change*100,2)}%')
       
        
    
    str_date = '01/13/23 10:00:00'
    
    start_date = datetime.strptime(str_date, '%m/%d/%y %H:%M:%S')
    
    tpcs = ['netflix_polarity','facebook_polarity','amazon_polarity','google_polarity','tesla_polarity']
    
    
    
    #open the sentiment csv file to get the information about the sentiment of each stocks generating by the twitter API and kafka
      
    df1=pd.read_csv("tweet_sentiment_netflix_polarity.csv")
    df2=pd.read_csv("tweet_sentiment_facebook_polarity.csv")
    df3=pd.read_csv("tweet_sentiment_amazon_polarity.csv")
    df4=pd.read_csv("tweet_sentiment_google_polarity.csv")
    df5=pd.read_csv("tweet_sentiment_tesla_polarity.csv")

    col1, col2 = st.columns(2)
    #construction of the price graph in real time.
    df_price_meta = yf.download(tickers='META',  start=start_date, interval='1m')
    df_price_netflix = yf.download(tickers='NFLX',  start=start_date, interval='1m')
    df_price_google = yf.download(tickers='GOOG',  start=start_date, interval='1m')
    df_price_amazon = yf.download(tickers='AMZN',  start=start_date, interval='1m')
    df_price_tesla = yf.download(tickers='TSLA.NE',  start=start_date, interval='1m')
    fig_meta = go.Figure(data=[go.Candlestick(x=df_price_meta.index[:1],
                open=df_price_meta['Open'][:1],
                high=df_price_meta['High'][:1],
                low=df_price_meta['Low'][:1],
                close=df_price_meta['Close'][:1])])
    fig_netflix = go.Figure(data=[go.Candlestick(x=df_price_netflix.index[:1],
                open=df_price_netflix['Open'][:1],
                high=df_price_netflix['High'][:1],
                low=df_price_netflix['Low'][:1],
                close=df_price_netflix['Close'][:1])])
    fig_amazon = go.Figure(data=[go.Candlestick(x=df_price_amazon.index[:1],
                open=df_price_amazon['Open'][:1],
                high=df_price_amazon['High'][:1],
                low=df_price_amazon['Low'][:1],
                close=df_price_amazon['Close'][:1])])
    fig_google = go.Figure(data=[go.Candlestick(x=df_price_google.index[:1],
                open=df_price_google['Open'][:1],
                high=df_price_google['High'][:1],
                low=df_price_google['Low'][:1],
                close=df_price_google['Close'])])
    fig_tesla = go.Figure(data=[go.Candlestick(x=df_price_tesla.index[:1],
                open=df_price_tesla['Open'][:1],
                high=df_price_tesla['High'][:1],
                low=df_price_tesla['Low'][:1],
                close=df_price_tesla['Close'][:1])])
    

    
   

    df_price_meta["return"]=df_price_meta["Close"].pct_change()
    df_prediction_meta = df_price_meta[["Open","High","Low","Close","Volume"]][1:]
    df_price_netflix["return"]=df_price_netflix["Close"].pct_change()
    df_prediction_netflix = df_price_netflix[["Open","High","Low","Close","Volume"]][1:]
    df_price_amazon["return"]=df_price_amazon["Close"].pct_change()
    df_prediction_amazon = df_price_amazon[["Open","High","Low","Close","Volume"]][1:]
    df_price_google["return"]=df_price_google["Close"].pct_change()
    df_prediction_google = df_price_google[["Open","High","Low","Close","Volume"]][1:]
    df_price_tesla["return"]=df_price_tesla["Close"].pct_change()
    df_prediction_tesla = df_price_tesla[["Open","High","Low","Close","Volume"]][1:]

    list_df_prediction=[df_prediction_meta,df_prediction_netflix,df_prediction_amazon,df_prediction_google,df_prediction_tesla]
    
    #find the minimum to stop the algo when we don't have anymore data
    minimum=len(df_prediction_meta)
    for i in list_df_prediction:
        if len(i)<minimum:
            minimum=len(i)

    low_bound_meta = np.percentile(df_price_meta["return"][1:], 35)
    high_bound_meta = np.percentile(df_price_meta["return"][1:], 65)
    low_bound_netflix = np.percentile(df_price_netflix["return"][1:], 35)
    high_bound_netflix = np.percentile(df_price_netflix["return"][1:], 65)
    low_bound_google = np.percentile(df_price_google["return"][1:], 35)
    high_bound_google= np.percentile(df_price_google["return"][1:], 65)
    low_bound_amazon = np.percentile(df_price_amazon["return"][1:], 35)
    high_bound_amazon = np.percentile(df_price_amazon["return"][1:], 65)
    low_bound_tesla = np.percentile(df_price_tesla["return"][1:], 35)
    high_bound_tesla = np.percentile(df_price_tesla["return"][1:], 65)
    
    list_high_bound=[low_bound_meta,low_bound_netflix,low_bound_google,low_bound_amazon,low_bound_tesla]
    list_low_bound=[high_bound_meta,high_bound_netflix,high_bound_google,high_bound_amazon,high_bound_tesla]
    # i: will represent the time here as we don't really analysis a stream
    i=0
    
    with col1:
        tab1, tab2, tab3 = st.tabs(["stock price","Positive Neutral Negative","polarity mean"])
        
        with tab1:
            st.markdown("META")
            pl_meta = st.empty()
            st.markdown("NETFLIX")
            pl_netflix = st.empty()
            st.markdown("AMAZON")
            pl_amazon = st.empty()
            st.markdown("GOOGLE")
            pl_google = st.empty()
            st.markdown("TESLA")
            pl_tesla = st.empty()
            pl_meta.plotly_chart(fig_meta,use_container_width=True)
            pl_amazon.plotly_chart(fig_amazon,use_container_width=True)
            pl_google.plotly_chart(fig_google,use_container_width=True)
            pl_tesla.plotly_chart(fig_tesla,use_container_width=True)
            pl_netflix.plotly_chart(fig_netflix,use_container_width=True)
        
        with tab2:
            df1['Sentiment'] = df1['polarity'].apply(getSentiment)
            df2['Sentiment'] = df2['polarity'].apply(getSentiment)
            df3['Sentiment'] = df3['polarity'].apply(getSentiment)
            df4['Sentiment'] = df4['polarity'].apply(getSentiment)
            df5['Sentiment'] = df5['polarity'].apply(getSentiment)
            count_neutral={'netflix': [((df1['Sentiment'].value_counts())["Neutral"])],'facebook': [(df2['Sentiment'].value_counts())["Neutral"]],'amazon': [(df3['Sentiment'].value_counts())["Neutral"]],'google': [(df4['Sentiment'].value_counts())["Neutral"]],'tesla': [(df5['Sentiment'].value_counts())["Neutral"]]}
            count_positive={'netflix': [(df1['Sentiment'].value_counts())["Positive"]],'facebook': [(df2['Sentiment'].value_counts())["Positive"]],'amazon': [(df3['Sentiment'].value_counts())["Positive"]],'google': [(df4['Sentiment'].value_counts())["Positive"]],'tesla': [(df5['Sentiment'].value_counts())["Positive"]]}
            count_negative={'netflix': [(df1['Sentiment'].value_counts())["Negative"]],'facebook': [(df2['Sentiment'].value_counts())["Negative"]],'amazon': [(df3['Sentiment'].value_counts())["Negative"]],'google': [(df4['Sentiment'].value_counts())["Negative"]],'tesla': [(df5['Sentiment'].value_counts())["Negative"]]}
    
            st.markdown("number of neutral tweet")
            neutral_graph=st.line_chart(count_neutral)
            st.markdown("number of positive tweet")
            positive_graph=st.line_chart(count_positive)
            st.markdown("number of negative tweet")
            negative_graph=st.line_chart(count_negative)
            
            
        with tab3:
            st.markdown("polarity graph")
            df_mean={'netflix': [(df1["polarity"]).mean()],'facebook': [(df2["polarity"]).mean()],'amazon': [(df3["polarity"]).mean()],'google': [(df4["polarity"]).mean()],'tesla': [(df5["polarity"]).mean()]}
            polarity_graph=st.line_chart(df_mean)
            
            
        
           
    with col2:
        
        st.markdown("prediction graph")
  
        knn_graph=st.line_chart({"META":[0.]})
        
    while True:
        
        #This is the part were we update all the previous defined graphs and metrics. Then we will see the evolution of each data that we have plotted in real time.
        
        
        if i!=0:
          
            price_meta=(df_price_meta.iloc[i])["Close"]
            price_netflix=(df_price_netflix.iloc[i])["Close"]
            price_amazon=(df_price_amazon.iloc[i])["Close"]
            price_google=(df_price_google.iloc[i])["Close"]
            price_tesla=(df_price_tesla.iloc[i])["Close"]
            change_meta=(price_meta-(df_price_meta.iloc[i-1])["Close"])/(df_price_meta.iloc[i-1])["Close"]
            change_netflix=(price_netflix-(df_price_netflix.iloc[i-1])["Close"])/(df_price_netflix.iloc[i-1])["Close"]
            change_amazon=(price_amazon-(df_price_amazon.iloc[i-1])["Close"])/(df_price_amazon.iloc[i-1])["Close"]
            change_google=(price_google-(df_price_google.iloc[i-1])["Close"])/(df_price_google.iloc[i-1])["Close"]
            change_tesla=(price_tesla-(df_price_tesla.iloc[i-1])["Close"])/(df_price_tesla.iloc[i-1])["Close"]
            m1.metric('META', f'{round(price_meta,1):,}', f'{round(change_meta*100,2)}%')
            m2.metric('NETFLIX', f'{round(price_netflix,1):,}', f'{round(change_netflix*100,2)}%')
            m3.metric('AMAZON', f'{round(price_amazon,1):,}', f'{round(change_amazon*100,2)}%')
            m4.metric('GOOGLE', f'{round(price_google,1):,}', f'{round(change_google*100,2)}%')
            m5.metric('TESLA', f'{round(price_tesla,1):,}', f'{round(change_tesla*100,2)}%')
        fig_meta = go.Figure(data=[go.Candlestick(x=df_price_meta.index[:i+1],
                    open=df_price_meta['Open'][:i+1],
                    high=df_price_meta['High'][:i+1],
                    low=df_price_meta['Low'][:i+1],
                    close=df_price_meta['Close'][:i+1])])
        fig_netflix = go.Figure(data=[go.Candlestick(x=df_price_netflix.index[:i+1],
                    open=df_price_netflix['Open'][:i+1],
                    high=df_price_netflix['High'][:i+1],
                    low=df_price_netflix['Low'][:i+1],
                    close=df_price_netflix['Close'][:i+1])])
        fig_amazon = go.Figure(data=[go.Candlestick(x=df_price_amazon.index[:i+1],
                    open=df_price_amazon['Open'][:i+1],
                    high=df_price_amazon['High'][:i+1],
                    low=df_price_amazon['Low'][:i+1],
                    close=df_price_amazon['Close'])])
        fig_google = go.Figure(data=[go.Candlestick(x=df_price_google.index[:i+1],
                    open=df_price_google['Open'][:i+1],
                    high=df_price_google['High'][:i+1],
                    low=df_price_google['Low'][:i+1],
                    close=df_price_google['Close'][:i+1])])
        fig_tesla = go.Figure(data=[go.Candlestick(x=df_price_tesla.index[:i+1],
                    open=df_price_tesla['Open'][:i+1],
                    high=df_price_tesla['High'][:i+1],
                    low=df_price_tesla['Low'][:i+1],
                    close=df_price_tesla['Close'][:i+1])])

        
        pl_meta.plotly_chart(fig_meta,use_container_width=True)
        pl_amazon.plotly_chart(fig_amazon,use_container_width=True)
        pl_google.plotly_chart(fig_google,use_container_width=True)
        pl_tesla.plotly_chart(fig_tesla,use_container_width=True)
        pl_netflix.plotly_chart(fig_netflix,use_container_width=True)
        
        df1=pd.read_csv('/home/quentin/.config/spyder-py3/tweet_sentiment_netflix_polarity.csv')
        df2=pd.read_csv("/home/quentin/.config/spyder-py3/tweet_sentiment_facebook_polarity.csv")
        df3=pd.read_csv("/home/quentin/.config/spyder-py3/tweet_sentiment_amazon_polarity.csv")
        df4=pd.read_csv("/home/quentin/.config/spyder-py3/tweet_sentiment_google_polarity.csv")
        df5=pd.read_csv("/home/quentin/.config/spyder-py3/tweet_sentiment_tesla_polarity.csv")
        
        
        df1['Sentiment'] = df1['polarity'].apply(getSentiment)
        df2['Sentiment'] = df2['polarity'].apply(getSentiment)
        df3['Sentiment'] = df3['polarity'].apply(getSentiment)
        df4['Sentiment'] = df4['polarity'].apply(getSentiment)
        df5['Sentiment'] = df5['polarity'].apply(getSentiment)
 
        count_neutral={'netflix': [(df1['Sentiment'].value_counts())["Neutral"]],'facebook': [(df2['Sentiment'].value_counts())["Neutral"]],'amazon': [(df3['Sentiment'].value_counts())["Neutral"]],'google': [(df4['Sentiment'].value_counts())["Neutral"]],'tesla': [(df5['Sentiment'].value_counts())["Neutral"]]}
        count_positive={'netflix': [(df1['Sentiment'].value_counts())["Positive"]],'facebook': [(df2['Sentiment'].value_counts())["Positive"]],'amazon': [(df3['Sentiment'].value_counts())["Positive"]],'google': [(df4['Sentiment'].value_counts())["Positive"]],'tesla': [(df5['Sentiment'].value_counts())["Positive"]]}
        count_negative={'netflix': [(df1['Sentiment'].value_counts())["Negative"]],'facebook': [(df2['Sentiment'].value_counts())["Negative"]],'amazon': [(df3['Sentiment'].value_counts())["Negative"]],'google': [(df4['Sentiment'].value_counts())["Negative"]],'tesla': [(df5['Sentiment'].value_counts())["Negative"]]}
        neutral_graph.add_rows(count_neutral)
        positive_graph.add_rows(count_positive)
        negative_graph.add_rows(count_negative)
        
        
        df_mean_new={'netflix': [(df1["polarity"]).mean()],'facebook': [(df2["polarity"]).mean()],'amazon': [(df3["polarity"]).mean()],'google': [(df4["polarity"]).mean()],'tesla': [(df5["polarity"]).mean()]}
        polarity_graph.add_rows(df_mean_new)
        
       
        if i<minimum:
            #This is the part were we compute the prediction of the movement of the facebook stock in real time.
            #We will not give a lot of details about this part as it has been already presented in the notebook of the project.
            #The classes are possible 0,1,2. In practice, trader will sell the stock when the algo predicts 0, buy the stock whenit predicts 2 and nothing when it predicts 1.
            #The only difference is that we used in this case the sentiment analysis provided by the twitter API to help us predict the futur movement of the price.
            
            
            dict_accuracy_knn={}
         
            global x_pred_meta
                

            
            
            x=dict(df_prediction_meta.iloc[i])
            
            if len(list_mm_meta)>n_wait:
                list_mm_meta.pop(0)
            list_mm_meta.append(x["Close"])
            x["MM_Close"] = np.mean(list_mm_meta)
          
            
            if len(list_mm_meta)==1:
              x_scaled = scaler.learn_one(x).transform_one(x)
            
            if len(list_mm_meta)>1:  
              y_true_reg = (x["Close"] - x_pred_meta["Close"]) /  x_pred_meta["Close"]
              for name, model in models.items():
                y_pred_reg = model.predict_one(x)
                x_pred_meta[name] = y_pred_reg
                model.learn_one(x,y_true_reg)
              x_scaled = scaler.learn_one(x_pred_meta).transform_one(x_pred_meta)  
              
              
              
              if (x["Close"] - x_pred_meta["Close"]) /  x_pred_meta["Close"] <low_bound_meta :
                  
                  if ((df2["polarity"]).mean())<-param:
                      y=0
                  
                  if ((df2["polarity"]).mean())>-param and ((df2["polarity"]).mean())<param :
                      y=1
                  if ((df2["polarity"]).mean())>param:
                      y=1
                      
                      
              if (x["Close"] - x_pred_meta["Close"]) /  x_pred_meta["Close"] > low_bound_meta and (x["Close"] - x_pred_meta["Close"]) /  x_pred_meta["Close"] < high_bound_meta :
                  
                  y=1
              if (x["Close"] - x_pred_meta["Close"]) /  x_pred_meta["Close"] > high_bound_meta :
                  if ((df2["polarity"]).mean())<-param:
                      y=1
                  
                  if ((df2["polarity"]).mean())<-param:
                      y=1
                  if ((df2["polarity"]).mean())>param:
                      y=2
                 
                  
              tab_y_true_meta.append(y)
              y_pred_meta= knn.predict_one(x_scaled)
              tab_pred_meta.append(y_pred_meta)
              acc_meta.update(y_true=y, y_pred=y_pred_meta)
            
              knn.learn_one(x_scaled,y)       
            x_pred_meta = x                
            dict_accuracy_knn["META"]=[float(acc_meta.get())]

            knn_graph.add_rows(dict_accuracy_knn)
          
            
        i=i+1
       
            
        time.sleep(5)
        
        
            
                
                
                
                
                  
               
            
        
models = {
    'lower': make_model(alpha=0.05),
    'center': make_model(alpha=0.5),
    'upper': make_model(alpha=0.95)
}            
        
scaler = preprocessing.StandardScaler()      
knn = KNNClassifier(n_neighbors=10, window_size=60)

acc_meta = metrics.BalancedAccuracy()

tab_pred_meta = []
tab_y_true_meta = []
list_mm_meta = []
x_pred_meta = {}


param=0.2


n_wait=100
verbose=False     
             
Main()
