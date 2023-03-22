from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

app = Flask(__name__,template_folder='template')

df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")
df_fake["label"]=0
df_true["label"]=1
df_fake_manual_testing = df_fake.tail(10)
for i in range(23460,23470):
    df_fake.drop([i])
df_true_manual_testing = df_true.tail(10)
for i in range(21397,21406,-1):
    df_true.drop([i])
df_manual_testing=pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv("manual testing.csv")
df_marge = pd.concat([df_fake, df_true], axis =0 )
df = df_marge.drop(["title", "subject","date"], axis = 1)
df = df.sample(frac = 1)
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
def output_lable(n):
    if n == 0:
        return "Misleading News"
    elif n == 1:
        return "Not A Misleading News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)

    return f"LR Prediction:{output_lable(pred_LR)}"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = manual_testing(message)
        return render_template('index.html', prediction=pred)
        print(pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
