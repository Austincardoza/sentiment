import pandas as pd
import pickle

df = pd.read_csv('tests/new.csv')
#getting rid of null values
df = df.dropna()
import numpy as np
np.random.seed(34)
df1 = df.sample(frac = 0.3)
df1['sentiments'] = df1.rating.apply(lambda x: 0 if x in [1, 2] else 1)
X = df1['review']
y = df1['sentiments']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=143)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ctmTr, y_train)
knn_score = knn.score(X_test_dtm, y_test)
print("Results for KNN Classifier with CountVectorizer")
print(knn_score)
y_pred_knn = knn.predict(X_test_dtm)
# df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Tweets', 'sentiment'])
pickle.dump(knn,open('model.pkl','wb'))
