import pandas as pd
import streamlit as st
import numpy as np
from tensorflow import keras
import keras
from PIL import Image
import pickle
import tensorflow as tf
from keras.utils import pad_sequences

from keras import backend as K
import streamlit as st
from nltk.tokenize import word_tokenize

import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Input, LSTM, Bidirectional
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib,os
import spacy
nlp= spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('wordnet')
import string
import re 
from bs4 import BeautifulSoup

from pathlib import Path
import base64

nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')

            
with open(r'C:\Users\brill\OneDrive\Documents\DScourse\fakenews project\models\passive_aggresive_model.pkl', 'rb') as file:
    model_LR = pickle.load(file)

## load the copy of the dataset
df = pd.read_csv('news_articles.csv')

####

###predict 

image =  Image.open(r'C:\Users\brill\OneDrive\Documents\DScourse\fakenews project\FNews.jpg')
st.image(image, use_column_width = True)



st.cache(allow_output_mutation =True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data=f.read()
    return base64.b64encode(data).decode()

def set_bg(jpg_file):
    bin_str =get_base64(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        }
        </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

    
       








def rem_en(input_txt):
    words= input_txt.lower().split()
    noise_free_words =[word for word in words if word not in stop ]
    noise_free_text= " ".join(noise_free_words)
    return noise_free_text


def punctuation_removal(text):
    all_list =[char for char in text if char not in string.punctuation]
    clean_str = " ".join(all_list)
    return clean_str

stemmer=nltk.PorterStemmer()
def clean_text(text):
    text = " ".join([word.lower() for word in text if word not in string.punctuation])
    text= " ".join([word for word in text.split()if word not in (stop)])
    text= ' '.join(re.sub("(w+://S+)", " ", text).split())
    text=rem_en(text)
    tokens= re.split('\w+', text)
    text = punctuation_removal(text)
    text= " ".join([i for i in text if not i.isdigit()])
    text= text.lower()
    text=text.split()
    text=" ".join(text)
    return text


def main():
    set_bg('FNews.jpg')
    from PIL import Image
    #### 

    ###image
    st.header("Hi, welcome, to our fake news classifier")
    st.write("please enter your news")
    activity = ["prediction", "NLP"]
    choice =  st.sidebar.selectbox("select activity", activity)
    if choice == "prediction":
        st.info("you have selected prediction using ML")
    else :
        st.info("you have selected prediction using NLP")
    
    
news_text= st.text_area("enter news below", "enter text here")
   
word_index = {word: index for index, word in enumerate(df.columns[:-4])}
numerical_news = [word_index[word] for word in news_text.lower().split() if word in word_index]

## pad the numerical email so it has the same shape as the training data
padded_news = pad_sequences([numerical_news], maxlen= 41355 )


########
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)##tfvect
##tfid_X_train = tfvect.fit_transform(df["text"]).values.astype(str)
##tfid_X_test = tfvect.transform(X_test)
tfidfvect = pickle.load(open(r'C:\Users\brill\OneDrive\Documents\DScourse\fakenews project\models\logistic_regression_model.pkl', 'rb'))

def fake_news_det(news):
    tfid_X_train = tfidfvect.fit_transform(df['X_train'].values.astype('U'))
    tfid_X_test = tfidfvect.transform(df['X_test'].values.astype('U'))
    input_data = [news_text]
    vectorized_input_data = tfidfvect.transform(input_data).values.astype('U')
    prediction = model_LR.predict(vectorized_input_data)
    return prediction 
    
      ##return prediction
      ##prediction = model_LR.predict(vectorized_input_data)
      ##return prediction

if st.button('classify'):
    prediction = 'FAKE' if model_LR.predict(padded_news) >0.5 else 'REAL'
    if prediction ==0:
        st.write('Fake News')

    else:
        st.write('Real News')




##prediction = model_LR.predict(vectorized_input_data)
##if news_text !="":
##if st.button('classify'):
	##processed_input = convert_input(user_input)
	##prediction = model_LR.predict(vectorized_input_data)
	##if prediction.item() == 0:                                     
		
		##st.write("The news are fake ")
		
	##else:
		
		##st.write("real  ")
		



#################################################

##if st.button('classify'):
   ## prediction= model_LR.predict(padded_news)
    ##if prediction > 0.5:
                ##st.write('the news are fake')
    ##else:
                ##st.write('the news are real')
###########################################################

        ##input_data_reshaped1 =clean_text(news_text)
        ##st.write("input text after text processing: \n ", input_data_reshaped1)
        ##if model_choice == 'LOG_REG':


            ##loaded_model =pickle.load(open('models\\RF_model.pkl','rb'))
           
           
                ##st.markdown('<p style="font-family:sans-serif;color:Green,font-size:48 px;">the news is fake</p>')
            
                ##=('<p style="font-family:sans-serif;color:Red,font-size:48 px;">the news is not fake fake</p>')
                ##st.markdown(new_title,unsafe_allow_html=True)
    

##if __name__ == '__main__': 
    ##main()