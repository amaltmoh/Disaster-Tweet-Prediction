import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import re
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import pickle
import csv
import spacy

nlp = spacy.load("en_core_web_sm")

#file = pd.read_csv(r"C:\Users\kkbmu\OneDrive\Desktop\mini project\Code\disaster-tweet-classifier-userinterface\ui streamlit\isodataset.swift")

st.set_page_config(page_title='Disaster Tweet Detection',page_icon=':tada:',layout='wide')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained SVM model
with open('ensem.pkl', 'rb') as f:
    finalmodel = pickle.load(f)

stop_words = set(stopwords.words('english'))
stop_words.update(["like", "u", "û_", "amp"])
lemmatizer = WordNetLemmatizer()

#side bar menu----
with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Home","Search","Contact"],
        icons=["house","search","envelope"],
        menu_icon="cast",
        default_index=0,
        styles={
        "container": {"padding": "0!important", "background-color": "#000000"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin":"0px",
            "--hover-color": "#808085",
        },
        "nav-link-selected": {"background-color": "grey"},
    },
    )

#-------HOME------------------------------
if selected=="Home":
    st.title("   DTWEET   ")
    def load_lottieurl(url):
        r=requests.get(url)
        if r.status_code!=200:
            return None
        return r.json()
    lottie_coding=load_lottieurl("https://lottie.host/88d25083-780b-47cc-af94-d50c4d82e70c/8rvvxxrYA2.json")
    with st.container():
        left_column,right_column=st.columns(2)
        with right_column:
            st_lottie(lottie_coding,height=500,key="coding")


#--------SEARCH--------------------------    
if selected=="Search":
    st.title("DISASTER TWEET OR NOT")
    st.write("---")
    st.markdown(
    """
    <style>
        .stTextInput input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            margin-top: 6px;
            margin-bottom: 16px;
            color: white; 
            resize: vertical
            font-family: Arial, sans-serif;
            font-size: 16px; 
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

    # Load the TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Preprocess text using the same vectorizer instance
    def preprocess_text(text, tfidf_vectorizer):
        text = re.sub(r"http\S+|www\S+|https\S+|\b\d+\b|\W", ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text= text.strip() 
        text = text.lower()  # Convert text to lowercase
        tokens = word_tokenize(text)  # Tokenize the text
        doc = nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        filtered_tokens = [word for word in tokens if word not in stop_words and word not in locations]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        text_transformed = tfidf_vectorizer.transform([' '.join(lemmatized_tokens)])
        return text_transformed

    # Make prediction using the same vectorizer instance
    def predict(text_transformed):
        pred = finalmodel.predict(text_transformed)
        return pred[0]


    def extract_location(text):
    # Process the input sentence using SpaCy
        doc = nlp(text)
        # Extract named entities and filter for locations
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # "GPE" is the label for geopolitical entities
        return locations


    def get_emergency_numbers(loc_address):
        with open('isodataset.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Country'] == loc_address:
                    st.text("EMERGENCY NUMBERS:")
                    st.write(
                        f"Police  : {row['Police']}    \n"
                        f"Ambulance: {row['Ambulance']}    \n"
                        f"Fire: {row['Fire']}    "
                    )
                    return
            st.text("Emergency Contact not found!")









    # Create a text input for entering the tweet
    tweet_text = st.text_input("Enter the tweet:",placeholder="Search...")
    if st.button('Submit'):
        if tweet_text:
            loc_address = extract_location(tweet_text)
            text_transformed = preprocess_text(tweet_text, tfidf_vectorizer)
            prediction = predict(text_transformed)
            if prediction == 1:
                st.error('This is a disaster tweet.')
                if loc_address:
                    loc_address = loc_address[0].strip("[]").capitalize()
                    get_emergency_numbers(loc_address)
                else:
                    st.warning('Location not found.')
            else:
                st.success('This is not a disaster tweet.')
        else:
            st.warning('Please enter a tweet to predict.')

    
        # Look up emergency numbers for the given location 

   
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    local_css("style1.css")


#-------------CONTACT---------------------------
if selected=="Contact":
    # st.title(f"FEEDBACK")
    st.header("Get In Touch With Us!")
    st.write("##")
    contact_form="""
    <form action="https://formsubmit.co/amaltmohan06@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Type your message here" required></textarea>
     <button type="submit">Send</button>
    </form>
    """
    left_column,right_column=st.columns(2)
    with left_column:
        st.markdown(contact_form,unsafe_allow_html=True)
    with right_column:
        st.empty()

#use local css---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
        
local_css("style.css")



