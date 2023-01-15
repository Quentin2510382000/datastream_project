# Streaming application for predictions of markets stocks
This project is part of the lectures of Data Stream processing of the M2DS from IP Paris with École Polytechnique. 

The authors of this project are : Quentin Lesbre, Lounes Guergous and Xavier Brouty. 

In this project, the objective is to do a streaming application for predictions of markets stocks. 
We first started to do a batch regression to compare the results to the one from the online learning with River. 
We also did an application of Kafka to have data from Twitter and do some sentimental analysis on stocks markets. 

The notebook for the batch regression and classification is : Batch_Regression_and_Classification.ipynb

The notebook for the online regression and classification is : Online_regression_River.ipynb

The files for the application of Kafka and the twitter api are : financial_dashboard, sentiment_analysis.py, ingest-tweets-stocks.py, consumer-stock-graph.py, consumer-stock-clean-polarity.py

We have to notify that the file for the kafka application and the twitter api are only for the purpose of displaying graphics during the presentation with the streamlit APi.

The results are presented in the notebooks and also in the slides of the presentation. 

