import streamlit as st

st.set_page_config(
    page_title="Email Classification App",
    page_icon="logo.png",  # URL or emoji
)

import pickle as pk

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# UDF for :-
# 1. convert into lower care
# 2. Removing stop words and punctuations
# Ensure that necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Removing special characters
    a = []
    for i in text:
        if i.isalnum():
            a.append(i)

    text = a[:]
    a.clear()

    # Removing stopwords and punc..
    for i in text:
        if i not in stopwords.words('english') and string.punctuation:
            a.append(i)

    text = a[:]
    a.clear()

    # Stemming the words
    for i in text:
        a.append(ps.stem(i))
    
    return " ".join(a)


# Read pickle files
tfidf = pk.load(open('vectorizer.pkl','rb'))
my_model = pk.load(open('model.pkl','rb'))

# UI of Web app
st.title("Email Spam Classifier")
input_email = st.text_input("Enter Your Email")

if st.button("Check"):
    # 1. Preprocess
    transform_email = transform_text(input_email)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_email])

    # 3. Predict
    result = my_model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")