import flask
from flask import * 
import pickle 
import nltk
import string
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def pre_processing(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[0-9]+','num',text)
    word_list = nltk.word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(item) for item in word_list]
    return ' '.join(word_list)


nltk_stopwords = stopwords.words('english')

wordnet_lemmatizer = WordNetLemmatizer()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=pre_processing,
                             ngram_range=(1, 3))

app = Flask(__name__ ,static_url_path='/static') 
model =pickle.load(open('quora_model.pkl','rb'))

@app.route('/') 
def Home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])


def predict():
    if request.method == "POST":
        input_on_button = request.form.to_dict()
        evaluate_test_question = vectorizer.fit_transform([input_on_button['question_text']])
        for evaluate_test_question in range (1483952):
            evaluate_test_question=[evaluate_test_question]
            j=np.array(evaluate_test_question)
            h=j.reshape((-1, 1))
            df=pd.DataFrame(h)
            df.columns=['question_text']
            #df.fillna((-999), inplace=True)
            #p=df.iloc[:,-1]
            prediction = model.predict(df[['question_text']])
            pred = prediction[0]
            out ="error"
            if pred ==1 :out ="insincere question"
            else :out ="sincere question"
            return render_template('index.html',results=out)
        else:
            return render_template('index.html')  

# def predict():

#     if request.method == "POST":
#         question_text = flask.request.form['question_text']
#         input_variables = pd.DataFrame([[question_text]],
#                                        columns=['question_text'])
#         prediction = model.predict(input_variables)[0]
#         pred = prediction[0]
#         out ="error"
#         if pred ==1 :out ="insincere question"
#         else :out ="sincere question"
#         return render_template('index.html',results=out)
#     else:
#         return render_template('index.html')  



       
if __name__ == "__main__": 

    app.run(host= '0.0.0.0',debug = True)