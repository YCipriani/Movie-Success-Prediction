import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel

columns = ["num_critic_for_reviews", "director_facebook_likes", "gross", "num_voted_users", "cast_total_facebook_likes", "num_user_for_reviews",  "budget", "movie_facebook_likes",  "actor_1_facebook_likes", "actor_2_facebook_likes",  "actor_3_facebook_likes", "imdb_score"]

df = pd.read_csv("movie_metadata.csv",sep = ",")
df = df.drop(columns = ["color", "title_year", "genres","duration", "movie_title", "director_name", "language", "country", "aspect_ratio", "content_rating", "movie_imdb_link","plot_keywords", "facenumber_in_poster", "actor_3_name", "actor_1_name",  "actor_2_name"])
decimals = 0
for col in columns:
	#df[col] = df[col].apply(lambda x: round(x, decimals))
	df[col] = df[col].fillna(0.0).astype(int)
print(df.head())

X = df.values[: ,0:10]
Y = df.values[:,11]
print(Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 40) 
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100) 

# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=4, min_samples_leaf=5)
# clf_gini.fit(X_train, y_train)
# y_pred = clf_gini.predict(X_test)
# print "Accuracy of Gini is ", accuracy_score(y_test,y_pred)*100

# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=4, min_samples_leaf=5)
# clf_entropy.fit(X_train, y_train)
# y_pred_en = clf_entropy.predict(X_test)
# print "Accuracy of Entropy is ", accuracy_score(y_test,y_pred_en)*100

# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X_train, y_train, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_transformed = scaler.transform(X_train)
# clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
# X_test_transformed = scaler.transform(X_test)
# print(clf.score(X_test_transformed, y_test))

#print "Accuracy of SVM is ", clf.score(X_test, y_test)


# print(x_train.values)
# print(y_train.values)
# print(type(x_train))

# x_train = x_train.fillna(0)
# y_train = y_train.fillna(0)

# y_train = y_train.replace(to_replace  = "B", value = 0)
# y_train = y_train.replace(to_replace  = "G", value = 1)
# print(x_train.values)
# print(y_train.values)



lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

clf = SVC(gamma='auto')
clf.fit(X_train, y_train) 
pred = clf.predict(X_test)
print "Accuracy of SVC is ", accuracy_score(y_test,pred)*100

# Y_new = model.transform(y_test)
# print(clf.predict(y_test))

# dr = pd.read_csv("1528897205_8734903_test-fin.csv",sep = ",")
# x = dr.values

# X_train, X_test, Y_train, Y_test = train_test_split(X_new, y_train, test_size=0.33, random_state=42)

# print(x)


# XYZ_new = model.transform(x)

# labels = clf.predict(XYZ_new)
# with open('format.data','w') as fp: 
	# fp.writelines(["%s\n" % str(item) for item in labels])


