import csv
import numpy
import random
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def stemmed_words(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))


def show_potential_answers():
    i = 0
    while i < len(doc_sim) and i < 5:
        print(doc_sim[i][1])
        print(doc_sim[i][0])
        print(corpus[class_id][doc_sim[i][1]])
        print('-----')
        i += 1


def show_accuracy_scores():
    X_new_counts = countvect["all"].transform(X_test)
    X_new_tfidf = tftransformer["all"].transform(X_new_counts)
    predicted = classifier.predict(X_new_tfidf)
    print(confusion_matrix(y_test, predicted))
    print()
    print('Accuracy score: ')
    print(accuracy_score(y_test, predicted))
    print()
    print('f1 score (Question answering): ')
    print(f1_score(y_test, predicted, average='binary', pos_label='qna'))
    print()
    print('f1 score (Small talk): ')
    print(f1_score(y_test, predicted, average='binary', pos_label='chat'))
    print()


posmap = {
    'ADJ': wordnet.ADJ,
    'ADV': wordnet.ADV,
    'NOUN': wordnet.NOUN,
    'VERB': wordnet.VERB
}

# Setup
sb_stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()
numpy.random.seed(seed=32)

directories = {
    "qna": "qna_datasheet.csv",
    "chat": "chat_datasheet.csv"
}
# KEY = QUESTION | VALUE = ANSWER
corpus = {
    "qna": {},
    "chat": {}
}
questions = {
    "qna": [],
    "chat": []
}
data_label = []
with open(directories["qna"], newline='', encoding='UTF-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader, None)
    for row in csv_reader:
        if row[0] not in corpus["qna"]:
            corpus["qna"][row[0]] = []
        corpus["qna"][row[0]].append(row[1])
        questions["qna"].append(row[0])
        data_label.append("qna")

with open(directories["chat"], newline='', encoding='UTF-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader, None)
    for row in csv_reader:
        corpus["chat"][row[0]] = [row[1], row[2], row[3]]
        questions["chat"].append(row[0])
        data_label.append("chat")

all_questions = questions['qna'] + questions['chat']
countvect = {
    "qna": CountVectorizer(analyzer=stemmed_words, stop_words=stopwords.words('english')),
    "chat": CountVectorizer(analyzer=stemmed_words, stop_words=stopwords.words('english'))
}
tftransformer = {
    "qna": TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True),
    "chat": TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
}
tdm = {
    "qna": countvect['qna'].fit_transform(questions['qna']),
    "chat": countvect['chat'].fit_transform(questions['chat'])
}
tfidf = {
    "qna": tftransformer['qna'].fit_transform(tdm['qna']),
    "chat": tftransformer['chat'].fit_transform(tdm['chat'])
}
X_train, X_test, y_train, y_test = train_test_split(all_questions, data_label, stratify=data_label, test_size=0.30)
count_vect = CountVectorizer(analyzer=stemmed_words, stop_words=stopwords.words('english'))
countvect["all"] = count_vect
X_train_counts = countvect["all"].fit_transform(X_train)
tdm["all"] = X_train_counts
tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
tftransformer["all"] = tfidf_transformer
X_train_tf = tftransformer["all"].fit_transform(X_train_counts)
tfidf["all"] = X_train_tf

try:
    classifier = load('clf_chatbot.joblib')
except FileNotFoundError:
    classifier = LogisticRegression(random_state=0).fit(tfidf["all"], y_train)
    dump(classifier, 'clf_chatbot.joblib')

# show_accuracy_scores()

stop = False
name = ""
set_name = ['change my name', 'fix my name', "that's not my name", 'set my name', 'i want a different name',
            'call me another name']
get_name = ["what's my name", "what is my name", 'give me my name', 'what do you call me', 'call my name',
            'say my name', 'tell me my name']

name = input("Hi! What's your name?\n")
query = input("\nAsk me anything, " + name + "! If you ever want to stop, be sure to type 'STOP'!\n")
while not stop:
    if 'STOP' in query:
        print("Thank you for talking with me! Bye!")
        stop = True
    else:
        if any(substring in query for substring in set_name):
            name = input("\nWhat would you like your name to be?\n")
            query = input("\nYour name is now " + name + "! \nAnything else?\n\n")
        elif any(substring in query for substring in get_name):
            query = input("\nYour name is " + name + "! \nAnything else?\n\n")
        else:
            query = [query]
            processed_query = countvect["all"].transform(query)
            processed_query = tftransformer["all"].transform(processed_query)
            class_id = classifier.predict(processed_query)
            class_id = class_id[0]

            tfidf_array = tfidf[class_id].toarray()
            processed_query = countvect[class_id].transform(query)
            processed_query = tftransformer[class_id].transform(processed_query)
            processed_query_arr = processed_query.toarray()[0]
            doc_sim = []
            for i in range(len(questions[class_id])):
                with numpy.errstate(invalid='raise'):
                    try:
                        tuple = (
                        1 - spatial.distance.cosine(processed_query_arr, tfidf_array[i]), questions[class_id][i])
                    except FloatingPointError:
                        break

                if tuple[0] > 0.5:
                    doc_sim.append(tuple)
            doc_sim.sort(reverse=True)
            # show_potential_answers()
            if len(doc_sim) > 0:
                random_ans = random.randint(0, len(corpus[class_id][doc_sim[0][1]]) - 1)
                # print(doc_sim[0][1])
                # print(doc_sim[0][0])
                # print(corpus[class_id][doc_sim[0][1]][random_ans])
                # print()
                query = input(corpus[class_id][doc_sim[0][1]][random_ans] + "\n\n")
            else:
                query = input("I'm sorry, I don't know how to answer... Anything else?\n\n")
