import requests

get_blog = requests.get('url')
json_blog = get_blog.json()

if json_data['status'] == 'OK' :
	//json data iterate kardiyo 




import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split 
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.svm import SVC

data = pd.read_csv('Tweets.csv')
def sentiment(text):
    if text == 'negative':
        return 0
    else:
        return 1 
data['final_sentiment'] = data['airline_sentiment'].apply(sentiment)##Training sentiment received


def meaningful(text):
    words = word_tokenize(text.lower())
    stopwds = set(stopwords.words("english"))
    stopuncts = ["@" , "#" , "$" , "%" , "^" , "&" , "*" ,"( " , ")" ,":",";",",",".","?","/" ,"!"]
    stopuncts = set(stopuncts)
    final_stops = stopwds.union(stopuncts)
    meaningful_words = [str(w) for w in words if not w in final_stops]
    return (" ".join(meaningful_words))

data['imp_words'] = data['text'].apply(meaningful)
x_train,x_test,y_train,y_test= train_test_split(data['imp_words'],data['final_sentiment'],test_size = 0.2)
v = CountVectorizer()
train_features = v.fit_transform(x_train)
test_features = v.transform(x_test)


# clf = tree.DecisionTreeClassifier()
# clf.fit(train_features, y_train)


clf2 = MultinomialNB()
clf2.fit(train_features, y_train)



# clf3 = SVC()
# clf3.fit(train_features, y_train)


# temp = max[clf.score(test_features, y_test),
# 			clf2.score(test_features, y_test),
# 			clf3.score(test_features,y_test)]
count_1 =1
count_0=0
for i in y_predict:
	if i ==0:
		count_0 +=1
	else:
		count_1 +=1
sentiment =""
if count_1>count_0:
	sentiment = "Positive"
else:
	sentiment = "Negative"

post_sentiment = requests.post('url',data={'sentiment':sentiment})
