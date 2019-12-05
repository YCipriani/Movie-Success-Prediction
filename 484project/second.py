import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

columns = ["gross", "cast_total_facebook_likes",  "director_facebook_likes", "movie_facebook_likes","actor_1_facebook_likes", "actor_2_facebook_likes",  "actor_3_facebook_likes","num_user_for_reviews", "num_critic_for_reviews", "budget", "num_voted_users", "imdb_score"]
string_columns =  ["genres","director_name", "content_rating","plot_keywords"]

# v = open('movie_metadata.csv')
# r = csv.reader(v)
# row0 = r.next()
# row0.append('Category')

# for item in r:
    # float_item = float(item[27])
    # if float_item < 5.0:
        # item.append("Flop")
    # elif 5.0 < float_item < 8.0:
        # item.append("Hit")
    # else:
        # item.append("BlockBuster")
		


df = pd.read_csv("movie_metadata.csv",sep = ",")
df = df.drop(columns = ["aspect_ratio","movie_imdb_link", "facenumber_in_poster","movie_title","title_year","language", "color", "duration", "country", "actor_3_name", "actor_1_name",  "actor_2_name"], axis = 1)
le = LabelEncoder()
for col in string_columns:
	df[col] = le.fit_transform(df[col])
	df[col] = df[col].fillna(0.0).astype(int)
	print(col + "\t" + str(df[col].corr(df["imdb_score"])))
	
# decimals = 0
for col in columns:
	#df[col] = df[col].apply(lambda x: round(x, decimals))
	print(col + "\t" + str(df[col].corr(df["imdb_score"])))
	df[col] = df[col].fillna(0.0).astype(int)
print(df.head())

# X = df.values[: ,0:11]
# Y = df.values[:,12]

# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X, Y)
# np.set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# print(list(df))

# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 40) 

# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=4, min_samples_leaf=5)
# clf_gini.fit(X_train, y_train)
# y_pred = clf_gini.predict(X_test)
# print "Accuracy of Gini is ", accuracy_score(y_test,y_pred)*100


# X = df.values[: ,0:10]
# Y = df.values[:,11]
# print(Y)
# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 40) 

# le = LabelEncoder()
# for col in string_columns:
	# df[col] = le.fit_transform(df[col])
	
# for c in list(df):
	#df[col] = df[col].apply(lambda x: round(x, decimals))
	# df[c] = df[c].fillna(0.0).astype(int)
# print(df.head())

# X = df.values[: ,0:20]
# Y = df.values[:,21]
# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 40) 

# scaler = StandardScaler() 
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)  
# X_test = scaler.transform(X_test) 

# classifier = KNeighborsClassifier(n_neighbors=60)  
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# print "Accuracy of KNN is ", accuracy_score(y_test,y_pred)*100