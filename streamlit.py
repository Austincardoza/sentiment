import streamlit as st
import tweepy as tw
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pickle
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from bs4 import BeautifulSoup
import re
import string
import os
import wget
import requests
from PIL import Image
from textblob import TextBlob
header = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36 OPR/83.0.4254.70'
    }
consumer_key = 'OCgWzDW6PaBvBeVimmGBqdAg1'
consumer_secret = 'tBKnmyg5Jfsewkpmw74gxHZbbZkGIH6Ee4rsM0lD1vFL7SrEIM'
access_token = '1449663645412065281-LNjZoEO9lxdtxPcmLtM35BRdIKYHpk'
access_token_secret = 'FL3SGsUWSzPVFnG7bNMnyh4vYK8W1SlABBNtdF7Xcbh7a'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
classifier = pipeline('sentiment-analysis')
st.set_option('deprecation.showPyplotGlobalUse', False)

model=pickle.load(open('model.pkl','rb'))
def clasifier(text):
    input=np.array([[text]]).astype(np.float64)
    prediction=model.predict(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return (pred)
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def run_twitter():
    st.title('Twitter')

    with st.form(key='Enter name'):
        search_words = st.text_input('Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 100 tweets)', 0,100,10)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
        tweet_list = [i.text for i in tweets]
        p = [i for i in classifier(tweet_list)]
        print(p)
        q=[p[i]['label'] for i in range(len(p))]
        r=[p[i]['score'] for i in range(len(p))]

        df = pd.DataFrame(list(zip(tweet_list, q,r)),columns =['Tweets', 'sentiment','accuracy'])
        st.write(df)
        
        title_type = df.groupby('sentiment').agg('count')
        print(title_type)
        #piechart
        piechart=df.sentiment.value_counts().plot(kind='pie',autopct="%1.0f%%")
        st.write(piechart)
        st.pyplot()

        Tweet_Texts=df['Tweets'].values
        Tweets_String=str(Tweet_Texts)
        #wordcloud
        import re
        Tweet_Texts_Cleaned = Tweets_String.lower()
        Tweet_Texts_Cleaned=re.sub(r'@\w+', ' ', Tweet_Texts_Cleaned)
        Tweet_Texts_Cleaned=re.sub(r'http\S+', ' ', Tweet_Texts_Cleaned)
        Tweet_Texts_Cleaned = re.sub(r'[^a-z A-Z]', ' ',Tweet_Texts_Cleaned)
        Tweet_Texts_Cleaned= re.sub(r'\b\w{1,2}\b', '', Tweet_Texts_Cleaned)
        Tweet_Texts_Cleaned= re.sub(r' +', ' ', Tweet_Texts_Cleaned)
        wordcloudimage = WordCloud(
                          max_words=100,
                          max_font_size=500,
                          font_step=2,
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=1000,
                          height=720
                          ).generate(Tweet_Texts_Cleaned)
        plt.imshow(wordcloudimage, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()
        
        csv = convert_df(df)

        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
        
def run_youtube():
    data=st.file_uploader("upload a file")
    if data is not  None:
        df=pd.read_csv(data)
        df['polarity'].replace(
        to_replace=['Neutral'],
        value='Negative',
        inplace=True
    )
        df.drop('subjectivity',
        axis='columns', inplace=True)
        st.write(df)
        title_type = df.groupby('polarity').agg('count')
        piechart=df.polarity.value_counts().plot(kind='pie',autopct="%1.0f%%")
        st.write(piechart)
        st.pyplot()

        histogram=plt.hist(df['Precision'], bins = 5)
        plt.xlabel('sentiment_score')
        st.write(histogram)
        st.pyplot()


loaded_model = pickle.load(open("amazon_train/finalized_model.sav", 'rb'))
def Amazon():
    
    st.title('Amazon')
    with st.form(key='Enter name'):
        term = st.text_input('Enter the name of product')
        no = st.number_input('Enter the number of products you want the sentiment for', 0,100,10)
        no_per = st.number_input('Number of reviews per product', 0,100,10)
        st.write("Searching for ",term)
        st.write("Number of Products ",no)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        val = loaded_model.predict(["This is bad"])
        records = []
        results = []
        rev = []    
        rev1=[]
        records_name = []
        def get_url(search_term):
            template = 'https://www.amazon.in/s?k={}&crid=2U21CDDK65IB9&ref=nb_sb_noss_2'
            search_term = search_term.replace(' ','+')
            return template.format(search_term)

        def get_asin(driver,no):
            soup = BeautifulSoup(driver.page_source,'html.parser')
            for i in soup.find_all('div',{'data-component-type':'s-search-result'})[:no]:
                records.append(i['data-asin'])
                records_name.append(i.h2.a.text.strip()) #================================heading============================
            for i in records:
                results.append(f"https://www.amazon.in/dp/product-reviews/{i}")
            for i in results:
                driver.get(i)
            reviews_unit(results)

        def clear_reviews(rev,dicton_reviews):
            for i in rev:
                text = i.lower()
                text = re.sub('\[.*?\]','',text)
                text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
                text = re.sub('\w*\d\w*','',text)
                text = re.sub(r'\b\w{1,2}\b', '', text)
                text = text.strip('\n')
                text = text.strip('\t')
                text=re.sub(r'@\w+', ' ', text)
                text=re.sub(r'http\S+', ' ', text)
                text = re.sub(r'[^a-z A-Z]', ' ',text)
                text= re.sub(r'\b\w{1,2}\b', '', text)
                text= re.sub(r' +', ' ', text)
                text = text.strip(" ")
                if text:
                    rev1.append(text)  

        dicton_reviews={}
        def reviews_unit(results):
            for j in range(len(results)):
                lst_rev = []
                page=requests.get(results[j],headers=header)
                soup=BeautifulSoup(page.content)
                for i in soup.findAll("span",{'data-hook':"review-body"})[:no_per]:

                    rev.append(i.text)
                    lst_rev.append(i.text)
                dicton_reviews[records_name[j]]=lst_rev
            # print(dicton_reviews)
            clear_reviews(rev,dicton_reviews) 



        def app(term,no,no_per):
            driver = webdriver.Edge(r'C:/Users/Hansel/Desktop/edgedriver_win64/msedgedriver.exe')
            # term =  st.text_input("Enter what you want:")
            # no = int(st.number_input("Enter the number of products:"))
            if term:
                url = get_url(term)
                driver.get(url)
            if no:
                get_asin(driver,no)
            elif no is 0 or no is None:
                st.write("Enter Proper Number of products.")
            # st.write(len(rev1))
            # st.table(rev1)
            else:
                st.write("Enter the product name")

            
            pol = {'0':'negative','1':'positive'}
            if rev1:
                # print("123",rev1)
                pos = 0
                neg = 0
                cloud = ""
                for i in rev1:
                    val = loaded_model.predict([i])
                    val = int(str(val)[1:2])
                    score = TextBlob(i).polarity
                    if (val-1)==1:
                        pos+=1
                    else:
                        neg+=1
                    st.write("REVIEW : " + str(i))
                    st.write("POLARITY : " + pol[str(val-1)])
                    cloud += i
                st.title("Word Cloud")
                wordcloudimage = WordCloud(
                            max_words=100,
                            max_font_size=500,
                            font_step=2,
                            stopwords=STOPWORDS,
                            background_color='black',
                            width=1000,
                            height=720
                            ).generate(cloud)
                plt.imshow(wordcloudimage, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
                st.title("Pie Chart")
                labels = 'Positive' , 'Negative'
                sizes = [pos,neg]
                explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)

        # df['sentiment'] = round(df['pos_count'] / (df['neg_count']+1), 2)
        app(term,no,no_per)


if __name__=='__main__':
    activities=['TWITTER','AMAZON','YOUTUBE']
    option=st.sidebar.selectbox('Selection option:',activities)
    if option=='TWITTER':
        run_twitter()
    if option=='AMAZON':
        Amazon()
    if option=='YOUTUBE':
        run_youtube()