import streamlit as st
import pickle as pkl
import nltk
import string
from nltk.stem.porter import PorterStemmer


stopwords = pkl.load(open('stopwords.pkl', 'rb'))
nltk.download('punkt')
def transform(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]

    y = []
    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)
    text = y[:]

    y = []
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    text = y[:]
    return " ".join(text)


tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

st.title("Message Spam Classifier")
sms = st.text_area("Enter the message")

if st.button('Classify'):

    trans = transform(sms)

    vect = tfidf.transform([trans])

    clas = model.predict(vect)
    if clas == 0:
        st.header("It is Not Spam.")
    else:
        st.header("It is Spam.")

