import matplotlib.pyplot as plt

import pandas as pd
# from pandas_ml import Confusion_Matrix

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 

# csv load and visualization
df = pd.read_csv('C:/Users/gorka/OneDrive/Escritorio/covid19_data.csv', index_col = 'ID')
print(df)

# Correlation
corrM = df.corr()
sns.heatmap(corrM, annot=True)
plt.show()

# Gaussian naive bayes (Multinomial? Complementary? ask)
def gaussian_nb_model():
	return GaussianNB()

# Support Vector Machine (SVM)
def svm_model():
	return SVC(kernel = 'linear', C = 1.0)

# Linear Regression
def linear_reg_model():
	return LinearRegression()

# Logistic Regression
def logistic_reg_model():
	return LogisticRegression()

# k-Nearest Neighbors
def knn_model():
	return KNeighborsClassifier()

class Model:
	gaussian_nb_model = gaussian_nb_model
	svm_model = svm_model
	linear_reg_model = linear_reg_model
	logistic_reg_model = logistic_reg_model
	knn_model = knn_model

def modeling(df, model: Model):
	x_train, x_test, y_train, y_test = train_test_split(df.drop(['EXITUS'], axis=1), df['EXITUS'], test_size = 0.25, random_state = 1)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	print(classification_report(y_test, y_pred))
	# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
	# df['y_Actual'] = df['y_Actual'].map({'Yes': 1, 'No': 0})
	# df['y_Predicted'] = df['y_Predicted'].map({'Yes': 1, 'No': 0})

	# confusion_matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])
	# confusion_matrix.print_stats()
	# sns.heatmap(confusion_matrix, annot = True)
	# plt.show()
