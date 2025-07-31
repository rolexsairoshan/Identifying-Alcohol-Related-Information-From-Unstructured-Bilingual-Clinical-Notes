from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime

import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Create your views here.
from Remote_User.models import ClientRegister_Model,identify_alcohol_information,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def predict_identify_alcohol_information_type(request):
    if request.method == "POST":

        Fid= request.POST.get('Fid')
        pid= request.POST.get('pid')
        gender= request.POST.get('gender')
        age= request.POST.get('age')
        hypertension= request.POST.get('hypertension')
        heart_disease= request.POST.get('heart_disease')
        clinical_note= request.POST.get('clinical_note')
        ever_married= request.POST.get('ever_married')
        work_type= request.POST.get('work_type')
        Residence_type= request.POST.get('Residence_type')
        avg_glucose_level= request.POST.get('avg_glucose_level')
        bmi= request.POST.get('bmi')
        smoking_status= request.POST.get('smoking_status')
        stroke= request.POST.get('stroke')


        df = pd.read_csv('Clinical_Datasets.csv')
        df
        df.columns

        # data under nlp
        print("Data Processing Under Natural Language Processing (NLP)")
        data = []
        Labels = []
        # Data Processing Under Natural Language Processing (NLP)
        for row in df["clinical_note"]:
            # tokenize words
            words = word_tokenize(row)
            # remove punctuations
            clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
            # remove stop words
            english_stops = set(stopwords.words('english'))
            characters_to_remove = ["''", '``', "rt", "https", "â€™", "â€œ", "â€", "\u200b", "--", "n't", "'s", "...",
                                    "//t.c"]
            clean_words = [word for word in clean_words if word not in english_stops]
            clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
            # Lematise words
            wordnet_lemmatizer = WordNetLemmatizer()
            lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
            data.append(lemma_list)

        df['label'] = df.alcohol_consumption.apply(lambda x: 1 if x == 1 else 0)
        df.head()

        cv = CountVectorizer()
        X = df['Fid']
        y = df['label']

        print("Fid")
        print(X)
        print("Label")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)


        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Alcohol Drinking Found'
        elif prediction == 1:
            val = 'Alcohol Drinking Found'

        print(val)
        print(pred1)

        identify_alcohol_information.objects.create(
        Fid=Fid,
        pid=pid,
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        clinical_note=clinical_note,
        ever_married=ever_married,
        work_type=work_type,
        Residence_type=Residence_type,
        avg_glucose_level=avg_glucose_level,
        bmi=bmi,
        smoking_status=smoking_status,
        stroke=stroke,
        Prediction=val)

        return render(request, 'RUser/predict_identify_alcohol_information_type.html',{'objs': val})
    return render(request, 'RUser/predict_identify_alcohol_information_type.html')



