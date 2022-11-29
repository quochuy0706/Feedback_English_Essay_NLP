import nltk
import pandas as pd
import numpy as np
from math import sqrt
import statistics
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

# Feature engineering
# Count number of characters
def count_chars(text):
    return len(text)
#Number of words
def count_words(text):
    return len(text.split())
#Number of capital words
def count_capital_words(text):
    return sum(map(str.isupper, text.split()))
#Number of punctuations
def count_punctuations(text):
    punctuations=",."
    d=dict()
    for i in punctuations:
        d[str(i)+' count']=text.count(i)
    return d
#Number of words in quotes
def count_words_in_quotes(phrase):
    count = {}
    for c in phrase:
        if not c.isalpha() and not c.isdigit() and c != "'":
            phrase = phrase.replace(c, " ")
    for word in phrase.lower().split():
        if word not in count:
            count[word] = 1
        else:
            count[word] += 1
    return count
#Number of sentences
def count_sent(text):
    return len(nltk.sent_tokenize(text))
#Count the number of unique words
def count_unique_words(text):
    return len(set(text.split()))
#Count of stopwords
def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)
#Sum quote and punct count
def count_special(dic):
    e = sum(dic.values())
    return e

df['char_count'] = df['full_text'].apply(lambda x:count_chars(x))
df['word_count'] = df['full_text'].apply(lambda x:count_words(x))
df['sent_count'] = df['full_text'].apply(lambda x:count_sent(x))
df['capital_word_count'] = df['full_text'].apply(lambda x:count_capital_words(x))
df['quoted_word_count'] = df['full_text'].apply(lambda x: count_words_in_quotes(x))
df['quoted_word_count'] = df['quoted_word_count'].apply(lambda x: count_special(x))
df['stopword_count'] = df['full_text'].apply(lambda x: count_stopwords(x))
df['unique_word_count'] = df['full_text'].apply(lambda x:count_unique_words(x))
df['punct_count'] = df['full_text'].apply(lambda x:count_punctuations(x))
df['punct_count'] = df['punct_count'].apply(lambda x: count_special(x))
df['avg_wordlength'] = df['char_count'] / df['word_count']
df['avg_sentlength'] = df['word_count'] / df['sent_count']
df['unique_vs_words'] = df['unique_word_count'] / df['word_count']
df['stopword_vs_words'] = df['stopword_count'] / df['word_count']

test['char_count'] = test['full_text'].apply(lambda x:count_chars(x))
test['word_count'] = test['full_text'].apply(lambda x:count_words(x))
test['sent_count'] = test['full_text'].apply(lambda x:count_sent(x))
test['capital_word_count'] = test['full_text'].apply(lambda x:count_capital_words(x))
test['quoted_word_count'] = test['full_text'].apply(lambda x: count_words_in_quotes(x))
test['quoted_word_count'] = test['quoted_word_count'].apply(lambda x: count_special(x))
test['stopword_count'] = test['full_text'].apply(lambda x: count_stopwords(x))
test['unique_word_count'] = test['full_text'].apply(lambda x:count_unique_words(x))
test['punct_count'] = test['full_text'].apply(lambda x:count_punctuations(x))
test['punct_count'] = test['punct_count'].apply(lambda x: count_special(x))
test['avg_wordlength'] = test['char_count'] / test['word_count']
test['avg_sentlength'] = test['word_count'] / test['sent_count']
test['unique_vs_words'] = test['unique_word_count'] / test['word_count']
test['stopword_vs_words'] = test['stopword_count'] / test['word_count']

column = df.columns.values.tolist()[2:8]
vectorizer = TfidfVectorizer()


# split the training and testing data set
full_text_vectorize = vectorizer.fit_transform(df['full_text']).toarray()
# Converting above list to DataFrame
full_text_df = pd.DataFrame(full_text_vectorize)
# Merging all features
Y = df[column]
column.extend(['full_text','text_id'])
feature_df = df.drop(columns=column, axis = 1)
X = pd.merge(full_text_df, feature_df ,left_index=True, right_index=True)

# Test Data
full_text_vectorize_test = vectorizer.transform(test['full_text']).toarray()
full_text_df_test = pd.DataFrame(full_text_vectorize_test)
y_feature_test = test.drop(columns=['text_id','full_text'], axis=1)
y_X_test = pd.merge(full_text_df_test,y_feature_test, left_index=True, right_index=True)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model Selection
#RMSE = {}
#model = [LinearRegression(), LGBMRegressor(), XGBRegressor()]
#for target in y_train.columns:
#    y_test_target = y_test[target]
#    y_train_target = y_train[target]
#    rms_list = []
#    for mo in model:
#        clf = mo.fit(X_train, y_train_target)
#        predicted = clf.predict(X_test)
#        rms = sqrt(mean_squared_error(y_test_target, predicted))
#        rms_list.append(rms)
#    RMSE[target] = rms_list
#print(RMSE)

# Hyper-parameter tuning
start = time.time()
parameters = {'num_leaves': [20,40,60,80,100], 'min_data_in_leaf': [5,10,15,20,25], 'max_depth':[-1,5,10,20],
              'learning_rate': [0.01, 0.05, 0.1, 0.2], 'reg_alpha':[0,0.01,0.03]}

best_para = {}
for target in y_train.columns:
    clf = RandomizedSearchCV(LGBMRegressor(), parameters,scoring='neg_root_mean_squared_error', n_iter=100, cv=5)
    clf.fit(X, Y[target])
    list_para = []
    list_para.extend([clf.best_params_,clf.best_score_,clf.best_estimator_])
    best_para[target] = list_para
end = time.time()
print(f'Time to excecute {(end-start)/3600} hours')

#with open('best_param.pkl', 'wb') as f:
#    pickle.dump(best_para, f)

best_model = {}
for target in best_para.keys():
    clf = LGBMRegressor().set_params(**best_para[target][0])
    model_target = clf.fit(X, Y[target])
    best_model[target] = model_target

#with open('best_model.pkl', 'wb') as f:
#    pickle.dump(best_model, f)

for target in best_model.keys():
    predicted = best_model[target].predict(y_X_test)
    test[target] = predicted

test_1 = test.drop(test.columns[[2,3,4,5,6,7,8,9,10,11,12,13]],axis=1)

test_1.to_csv('submission.csv')
