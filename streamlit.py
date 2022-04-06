import streamlit as st
import tweepy as tw
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pickle
import numpy as np


from PIL import Image

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
        q=[p[i]['label'] for i in range(len(p))]
        df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Tweets', 'sentiment'])
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
        
if __name__=='__main__':
    activities=['TWITTER','AMAZON','YOUTUBE']
    option=st.sidebar.selectbox('Selection option:',activities)
    if option=='TWITTER':
        run_twitter()
    if option=='AMAZON':
        st.write("amazon")
    if option=='YOUTUBE':
        st.write("youtube")