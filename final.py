import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import spacy

nlp = spacy.load("en_core_web_sm")

# Data is loaded
# Data is loaded
data1 = pd.read_csv(r"C:\Users\ASUS\Desktop\amal\train.csv")
data2= pd.read_csv(r"C:\Users\ASUS\Desktop\amal\train2.csv")
max_id = data1['id'].max()

# Reset the index of the second dataset and adjust its IDs
data2['id'] += max_id

# Combine the datasets
data = pd.concat([data1, data2], ignore_index=True)

'''Here URLs,non-alphanumeric characters and whitespaces are removed. The cleaned text is then tokenised. 
   Stop words are removed from those tokens and then lemmatised. lemmatised tokens are then joined and returned.'''

stop_words = set(stopwords.words('english'))
stop_words.update(["like", "u", "û_", "amp"])
lemmatizer = WordNetLemmatizer()

# preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+|\b\d+\b|\W", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text= text.strip() 
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in locations]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens) 
data['text'] = data['text'].apply(preprocess_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=43)

'''Feature extraction and vectorisation '''
tfidf_vectorizer = TfidfVectorizer(min_df=7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


'''logistic '''
print("LOGISTIC REGRESSION")
'''training the model'''
logistic_clf_tfidf = LogisticRegression()
#logistic_clf_tfidf = LogisticRegression(solver="liblinear")
logistic_clf_tfidf.fit(X_train_tfidf, y_train)

'''evaluation'''

#training accuracy
y_train_log = logistic_clf_tfidf.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_log)
print("Training Accuracy:{:.2f}".format(train_accuracy))

#testing accuracy
y_test_log = logistic_clf_tfidf.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_log)

print("Testing Accuracy : {:.2f}".format(test_accuracy))

# classification report for testing data
print("Classification Report for Testing Data:\n", classification_report(y_test, y_test_log))

# Save the TF-IDF vectorizer to a file using pickle
tfidf_vectorizer_path = r"C:\Users\ASUS\Desktop\amal\tfidf_vectorizer.pkl"
with open(tfidf_vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)



'''SVM'''
print("SVM")
from sklearn.svm import SVC

# Create an SVC model with class weights
svc_clf_tfidf = SVC()

# Train the model
svc_clf_tfidf.fit(X_train_tfidf, y_train)

'''Evaluation'''
# Training accuracy
y_train_svm = svc_clf_tfidf.predict(X_train_tfidf)
train_accuracy_svm = accuracy_score(y_train, y_train_svm)
print("Training Accuracy (SVM): {:.2f}".format(train_accuracy_svm))

# Testing accuracy
y_test_svm = svc_clf_tfidf.predict(X_test_tfidf)
test_accuracy_svm = accuracy_score(y_test, y_test_svm)
print("Testing Accuracy (SVM): {:.2f}".format(test_accuracy_svm))

# Print classification report for testing data
print("Classification Report for Testing Data (SVM):\n", classification_report(y_test, y_test_svm))





'''Random Forest'''

print("RANDOM FOREST")
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model with tuned parameters
rf_clf_tfidf = RandomForestClassifier()

# Train the model
rf_clf_tfidf.fit(X_train_tfidf, y_train)

'''Evaluation'''
# Training accuracy
y_train_rf = rf_clf_tfidf.predict(X_train_tfidf)
train_accuracy_rf = accuracy_score(y_train, y_train_rf)
print("Training Accuracy (Random Forest): {:.2f}".format(train_accuracy_rf))

# Testing accuracy
y_test_rf = rf_clf_tfidf.predict(X_test_tfidf)
test_accuracy_rf = accuracy_score(y_test, y_test_rf)
print("Testing Accuracy (Random Forest): {:.2f}".format(test_accuracy_rf))

# Print classification report for testing data
print("Classification Report for Testing Data (Random Forest):\n", classification_report(y_test, y_test_rf))





'''XGBOOST'''
import xgboost as xgb

# Training the XGBoost model
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train_tfidf, y_train)

# Evaluation

# Training accuracy
y_train_xgb = xgb_clf.predict(X_train_tfidf)
train_accuracy_xgb = accuracy_score(y_train, y_train_xgb)
print("Training Accuracy (XGBoost): {:.2f}".format(train_accuracy_xgb))

# Testing accuracy
y_test_xgb = xgb_clf.predict(X_test_tfidf)
test_accuracy_xgb = accuracy_score(y_test, y_test_xgb)
print("Testing Accuracy (XGBoost): {:.2f}".format(test_accuracy_xgb))

# Classification report for testing data
print("Classification Report for Testing Data (XGBoost):\n", classification_report(y_test, y_test_xgb))


'''ensemble'''

print("ENSEMBLE")
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define the meta-learner (classifier)
meta_learner = LogisticRegression()

# Create Stacking ensemble model
stacking_clf = StackingClassifier(estimators=[
    ('logistic', logistic_clf_tfidf),
    ('svm', svc_clf_tfidf),
    ('xgb', xgb_clf),
    ('random_forest', rf_clf_tfidf)
], final_estimator=meta_learner)

# Train the Stacking ensemble model
stacking_clf.fit(X_train_tfidf, y_train)

# Evaluation

# Training accuracy
train_accuracy_stacking = stacking_clf.score(X_train_tfidf, y_train)
print("Training Accuracy (Stacking): {:.2f}".format(train_accuracy_stacking))

# Testing accuracy
test_accuracy_stacking = stacking_clf.score(X_test_tfidf, y_test)
print("Testing Accuracy (Stacking): {:.2f}".format(test_accuracy_stacking))

# Classification report for testing data
y_test_pred_stacking = stacking_clf.predict(X_test_tfidf)
print("Classification Report for Testing Data (Stacking):\n", classification_report(y_test, y_test_pred_stacking))

ens_path=r"C:\Users\ASUS\Desktop\amal\ensem.pkl"
with open(ens_path, 'wb') as f:
    pickle.dump(stacking_clf,f)