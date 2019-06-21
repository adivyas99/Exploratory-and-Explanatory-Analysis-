# Exploratory analysis using Statistics and visualization
# Classification using all major supervised ML algorithms 

# Dataset used can be downloaded from-
https://www.kaggle.com/carlolepelaars/toy-dataset 

#-----------------------------------------------------
# IMPORTING LIBRARIES
#-----------------------------------------------------

# For number manipulation and management
import numpy as np

# For visualization
import matplotlib.pyplot as plt

# For data management
import pandas as pd

# For visualization
import seaborn as sns

# Plotting ROC with sklearn
import scikitplot as skplt

# For unbiased training
from sklearn.model_selection import cross_val_score

# For classification metrices
from sklearn.metrics import accuracy_score, precision_score, classification_report

# For statistical tools
from scipy import stats

# For encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#-----------------------------------------------------
# IMPORTING DATASETS
#-----------------------------------------------------

# Importing the dataset
dataset = pd.read_csv('toy_dataset.csv')

# Extracting features and labels-
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5:6].values

#-----------------------------------------------------
# DATA EXPLORATION-
#-----------------------------------------------------

# Check for NULL Values-
print(dataset.isnull().values.any()) # function of pandas as 'dataset' is of type pd.dataframe

# To check the dimension of our data
print(np.ndim(X)) #2D

# No of values in the 2D array
print(np.size(X))

# Check type
print(X.dtype)

# Info about datatype
print(np.info('float64'))

# Basic Descriptive statistics of data-
print(dataset.describe())
# See negative value in salary
# We should remove the Negative value

# Changing all negative values to in salary to positive-
dataset['Income'] = dataset['Income'].abs()

# Since we have numpy array 
# So to check the unique values we'll need to change them to pandas series-
No_of_unique_values_City = pd.value_counts(pd.Series(X[:, 0]))
print(No_of_unique_values_City)

No_of_unique_values_Gender = pd.value_counts(pd.Series(X[:, 1]))
print(No_of_unique_values_Gender)

No_of_unique_values_Age = pd.value_counts(pd.Series(X[:, 2]))
print(No_of_unique_values_Age)

No_of_unique_values_Income = pd.value_counts(pd.Series(X[:, 3]))
print(No_of_unique_values_Income)
# Certainly income is a continuous variable

#-----------------------------------------------------
# Encoding categorical data-
#-----------------------------------------------------

# LABEL ENCODER------->
# See the different columns and then apply the encoding to the required one-
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
print(labelencoder_X_1.classes_)
print(len(labelencoder_X_1.classes_)) # verified

labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
print(labelencoder_X_2.classes_)
print(len(labelencoder_X_2.classes_)) # verified

# To check back the original value-
print(labelencoder_X_1.inverse_transform([4]))
print(labelencoder_X_2.inverse_transform([1]))

#-------------------------------

# ONE HOT ENCODER------->
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# see all categories-
print(onehotencoder.categories_)
print(onehotencoder.active_features_)

# max no of values perfeature
print(onehotencoder.n_values_) # city has 8 unique values

# Lets print X-
print(X)
print(X.dtype)

# to see number of unique labels through pandas-
No_of_unique_values_Label = pd.value_counts(pd.Series(y[:, 0]))

# as we can see that its a string we'll need to change that to numbers with LabelEncoder-
labelencoder_y = LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])

# For y- which is our label--
print(y) 
print(y.dtype)  # its of object type, but we have transformed all the value to integers

# converting the Dtype of Y-
y = y.astype(int)

#-----------------------------------------------------
# Inferential statistics of data-
#-----------------------------------------------------
# Coverting Data into DF format-
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

# Pearson correlation ---

#Age and Salary
print(stats.pearsonr(X_df[9], X_df[10]))
# returns (R-value and P-value)

# Spearman Correlation between Age and Salary
print(X_df[9].corr(X_df[10],method= 'spearman'))

#-------------------------------

# Conduct a Chi-square test of independence between 2 categorical variables ---
# returns (ChiSquare Value, P-Value, Degree of Freedom, expected frequencies as an array)

# Age and Illness-
crosstab = pd.crosstab(X_df[9], y_df[0])
print(stats.chi2_contingency(crosstab))

# Salary and Illness-
crosstab = pd.crosstab(X_df[10], y_df[0])
print(stats.chi2_contingency(crosstab))

# Gender and Illness-
crosstab = pd.crosstab(X_df[10], y_df[0])
print(stats.chi2_contingency(crosstab))

#-----------------------------------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# 80% to train and 20% to test-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 39)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Don't need to scale Y as they are categorical and with only 2 unique numerical values-

#-----------------------------------------------------
# SUPERVISED MACHINE LEARNING ALGORITHMS ---
#-----------------------------------------------------

# Logistic regression (classifier)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting the points will give NOTHING-
# as they are 0's and 1's -- 
plt.scatter(y_test,y_pred)
plt.legend()

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))

# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 
# precision is worst as its 0!
# which is proven by confusion matrix

# Check the slope (coefficent) and intercept-
print(classifier.coef_ )  # 11 features-
print(classifier.intercept_)  # bias value added to the decision function
print(classifier.classes_) # No of output classes

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

#-----------------------------------------------------

# Support Vector Classifier (Linear kernel)-

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability=True) 
# probability = True neccesary for predict_prob
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# Check the slope (coefficent) and intercept-
print(classifier.coef_ )  # 11 features-
print(classifier.intercept_)  # bias value added to the decision function

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

#-----------------------------------------------------

# Support Vector Classifier (Rbf kernel)-

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# Check the intercept- NO slope as its kernel is rbf 
print(classifier.intercept_)  # bias value added to the decision function

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())
#-----------------------------------------------------

# Naive Bayes classifier-

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# No of training examples to see-
print(classifier.class_count_)
# probability of each class-
print(classifier.class_prior_)

# mean and Variance of each feature per class-
print(classifier.theta_)
print(classifier.sigma_)
# To know the absolute value added to the variance-
print(classifier.epsilon_)

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

#-----------------------------------------------------

# Decision Tree Classifier-

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 32)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# List pf importance of all features-
print(classifier.feature_importances_)
print(classifier.tree_)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

#-----------------------------------------------------

# Random Forest Clasifier-

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

# Change value of n_estimators - 1,10,30,50,100,200,500
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

# Applying Grid search Cross Validation
# Grid search to check for different hyperparameters in our model
from sklearn.grid_search import GridSearchCV

parameters = {
              "n_estimators": [1,30,100,200,500],
              "random_state": [0,30,42]
             }

grid = GridSearchCV(estimator=classifier,
                    param_grid=parameters)

grid.fit(X_train, y_train)

# print best accuracy-
print(grid.best_score_)
# Print best parameters-
print(grid.best_estimator_.n_estimators)

#-----------------------------------------------------

# XGBoost Classifier-

from xgboost import XGBClassifier

# Fitting XGBoost to the Training set
classifier = XGBClassifier(max_depth=5,n_estimators=100)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy of our model-
print(accuracy_score(y_test, y_pred))
# Precision of our model- 
# FORMULA - tp / (tp + fp)
print(precision_score(y_test, y_pred)) 

# To generate a classification report--
# Here recall is the ratio tp / (tp + fn) 
print(classification_report(y_test, y_pred))

# ROC curve--
y_prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_prob)
plt.show()

# Applying k-Fold Cross Validation
scores = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(scores.mean())
print(scores.std())

# Coefficient of Variation (CV)
# if CV>=1 then high variation else CV<1 low variation
print(scores.std()/scores.mean())

# Applying Random search Cross Validation
from sklearn.grid_search import RandomizedSearchCV

params = {
          "max_depth": [10,20,40,50],
          "n_estimators": [100,300,500]
         }

random_search = RandomizedSearchCV(estimator=classifier, 
                                   param_distributions=params, 
                                   cv=5, 
                                   n_iter=8, 
                                   random_state=5)

random_search.fit(X_train, y_train) 

# print best accuracy
print(random_search.best_score_)
# Print best parameters-
print(random_search.best_estimator_.max_depth)

#-----------------------------------------------------
# VISUAL ANALYSIS for Explainatory Analysis--
#-----------------------------------------------------

# Scatter plot between Income and Age to show data points-
# Distribution of data-
sns.scatterplot(x='Income',y='Age',data = dataset)

# Distribution of Ages and Cities
sns.catplot(x="Age",y="City",data=dataset)

# Volume Distribution of Ages and Cities
sns.catplot(x="Age",y="City",kind='violin',data=dataset)

# Count of Income with Illness in bar-graph
sns.catplot(x="Illness",y="Income",kind='bar',data=dataset)

# Count of Income with Illness in bar-graph with respect to cities
sns.catplot(x="Illness",y="Income",kind='bar',hue='City',data=dataset)

# Count of different cities with labels alligned
ax = sns.catplot(x='City',kind='count',data=dataset,orient="h")
ax.fig.autofmt_xdate()

# Avg age and Gender with respect to Cities
ax = sns.catplot(x='Gender',y='Age',hue='City',kind='point',data=dataset)
ax.fig.autofmt_xdate()

# Count of data- Ill or Not- in bar format
sns.countplot(x = 'Illness', data = dataset)

# Scattering different ages with respect to Income
plt.scatter(x = 'Age', y= 'Income', data = dataset)
# Similarly using MatplotLib-
plt.plot(dataset.Age,dataset.Income,ls='',marker='o',color="green")

# Relation between all variables with respect to Illness or not
sns.pairplot(dataset, hue='Illness',height=3)

# Line graph between Age and Salary
(dataset.groupby('Age')['Income'].mean().plot(fontsize=10.0,figsize=(5,5)))

# Age Distribution count using histogram
plt.hist(dataset.Age,bins=10)

# Income Distribution count whose are Ill using histogram
plt.hist(dataset[dataset['Illness']=='Yes'].Income, bins=10)

# Age Distribution count whose are Ill using histogram
plt.hist(dataset[dataset['Illness']=='Yes'].Age, bins=10)

# Proportion of Illness according to Cities using stacked bar chart
table=pd.crosstab(dataset.City,dataset.Illness)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of City vs Illness')
plt.xlabel('City')
plt.ylabel('Illness')
plt.savefig('Stacked City_vs_Illness')

# Count of Illness according to Cities using stacked bar chart
pd.crosstab(dataset.City,dataset.Illness).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('City')
plt.ylabel('Illness')
plt.savefig('Count City_vs_Illness')



######## THANK  YOU!!! #########

