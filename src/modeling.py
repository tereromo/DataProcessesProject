import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas_ml import Confusion_Matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Csv load and visualization

#df = pd.read_csv('COVID19_data.csv', index_col = 'ID')
#print(df)

#Ranges

#print(min(df.AGE))
#print(max(df.AGE))
#print(min(df.DAYS_HOSPITAL))
#print(max(df.DAYS_HOSPITAL))
#print(min(df.DAYS_ICU))
#print(max(df.DAYS_ICU))
#print(min(df.TEMP))
#print(max(df.TEMP))
#print(min(df.HEART_RATE))
#print(max(df.HEART_RATE))
#print(min(df.GLUCOSE))
#print(max(df.GLUCOSE))
#print(min(df.SAT_O2))
#print(max(df.SAT_O2))
#print(min(df.BLOOD_PRES_SYS))
#print(max(df.BLOOD_PRES_SYS))
#print(min(df.BLOOD_PRES_DIAS))
#print(max(df.BLOOD_PRES_DIAS))

# Stay and death stats

#print(mean(df.DAYS_HOSPITAL))
#print(mean(df.DAYS_ICU))
#df.EXITUS.describe()

# Correlation

#corrM = df.corr()
#sn.heatmap(corrM, annot=True)
#plt.show()


# Models

#################################################

# Gaussian naive bayes (Multinomial? Complementary? ask)

def gaussian_NB_model():
	return GaussianNB()

#################################################

# Support Vector Machine (SVM)

def svm_model():
	return SVC(kernel = 'linear', C = 1.0)

#################################################

# Linear Regression

def linear_reg_model():
	return LinearRegression()

#################################################

# Logistic Regression

def logistic_reg_model():
	return LogisticRegression()

#################################################

# k-Nearest Neighbors

def knn_model():
	return KNeighborsClassifier()

#################################################

class Model:
	gaussian_NB_model = gaussian_NB_model
	svm_model = svm_model
	linear_reg_model = linear_reg_model
	logistic_reg_model = logistic_reg_model
	knn_model = knn_model

def modeling(df, model: Model):
	X_train, X_test, y_train, y_test = train_test_split(df.drop(['EXITUS'], axis=1),df['EXITUS'], test_size = 0.25, random_state = 1)
	md = model()
	md.fit(x_train, y_train)
	y_pred = md.predict(x_test)

	print(classification_report(y_test, y_pred))
	df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
	df['y_Actual'] = df['y_Actual'].map({'Yes': 1, 'No': 0})
	df['y_Predicted'] = df['y_Predicted'].map({'Yes': 1, 'No': 0})

	confusion_matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])
	confusion_matrix.print_stats()
	sn.heatmap(confusion_matrix, annot = True)
	plt.show()
