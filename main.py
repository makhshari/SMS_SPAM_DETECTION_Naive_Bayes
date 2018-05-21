import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")



def calculate_params(result) :
    params_array=[]
    params_array["TP"]=0
    params_array["FP"]=0
    params_array["TN"]=0
    params_array["FN"]=0

    return params_array

def calculate_metrics(result):
    metrics_array=[]
    params_array=calculate_params(result)

    TP = params_array["TP"]
    FP = params_array["FP"]
    TN = params_array["TN"]
    FN = params_array["FN"]

    P = TP + FN
    N = TN + FP
    total=TP+FP+TN+FN

    metrics_array["Accuracy"]=(TP+TN)/(P+N)
    metrics_array["Precision"] = TP/(TP+FP)
    metrics_array["Recall"] = TP/P

    return metrics_array

def analysis (dataset):
    print(dataset.describe())
    count_Class = pd.value_counts(dataset["label"] , sort=True)
    #drawing Bar Chart
    # count_Class.plot(kind='bar', color=["blue", "red"])
    # plt.title('email spam collection bar')
    # plt.show()

    # drawing Pie Chart
    # count_Class.plot(kind='pie', autopct='%1.0f%%')
    # plt.title('email spam collection Pie chart')
    # plt.ylabel('')
    # plt.show()

    # Showing the spam & ham words
    ham_words = ''
    spam_words = ''
    spam = dataset[dataset.spam == 1]
    ham = dataset[dataset.spam == 0]
    for val in spam.content:
        content = val.lower()
        tokens = nltk.word_tokenize(content)
        # tokens : the words that does not exist in the stopwords list
        for words in tokens:
            spam_words = spam_words + words + ' '

    for val in ham.content:
        content = val.lower()
        tokens = nltk.word_tokenize(content)
        for words in tokens:
            ham_words = ham_words + words + ' '
    spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
    plt.figure( figsize=(10,8), facecolor='k')
    plt.imshow(spam_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return


def NBC_prediction(trainData_df,y_train):
    prediction = dict()
    model = MultinomialNB()
    model.fit(trainData_df, y_train)
    prediction = model.predict(testData_df)
    return prediction

def show_metrics(y_test,NBC_predicted) :
    print("Accuracy : ",accuracy_score(y_test,NBC_predicted))
    print("Recall : ",recall_score(y_test,NBC_predicted , average='macro'))
    print("Precision : ",precision_score(y_test,NBC_predicted,average='macro'))
    return


#read data from file
dataset = pd.read_csv('./spam.csv',usecols=[0,1],encoding='latin-1')
dataset.columns = ["label","content"]
# build a binary column for detecting spam
dataset['spam'] = dataset.label.map({'ham': 0, 'spam': 1})
#analysis(dataset)

#test and train Split ---> "content" is the feature and "label" is the target
trainData,testData,y_train,y_test = train_test_split(dataset["content"],dataset["label"], test_size = 0.01, random_state = 10)

vect = CountVectorizer()
# learn the vocabulary
vect.fit(trainData)
#transformed type of train and test data
trainData_df = vect.transform(trainData)
testData_df = vect.transform(testData)
NBC_predicted = NBC_prediction(trainData_df,y_train)
show_metrics(y_test,NBC_predicted)
prediction = pd.DataFrame(NBC_predicted, columns=['predicted by MultinomialNB']).to_csv('prediction.csv')


