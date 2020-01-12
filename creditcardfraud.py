import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix,roc_curve,auc

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('creditcard.csv')
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

print(df['Class'].value_counts() )      # 0 - NonFraud Class
                                        # 1 - Fraud Class

# to get in percentage use 'normalize = True'
print('\nNoFrauds = 0 | Frauds = 1\n')
print(df['Class'].value_counts(normalize = True)*100)
df['Class'].value_counts().plot(kind = 'bar', title = 'Class Distribution\nNoFrauds = 0 | Frauds = 1');
plt.show()

import seaborn as sns
fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, figsize=(20, 5))

ax1.set_title(' Variable V1-V28\nAssuming as Scaled')  # plotting only few variables
sns.kdeplot(df['V1'], ax=ax1)                          # kde - kernel density estimate
sns.kdeplot(df['V2'], ax=ax1)
sns.kdeplot(df['V3'], ax=ax1)
sns.kdeplot(df['V25'], ax=ax1)
sns.kdeplot(df['V28'], ax=ax1)

ax2.set_title('Time Before Scaling')
sns.kdeplot(df['Time'], ax=ax2)

ax3.set_title('Amount Before Scaling')
sns.kdeplot(df['Amount'], ax=ax3)

plt.show()

from sklearn.preprocessing import StandardScaler,RobustScaler
rb = RobustScaler()
df['Time'] = rb.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount'] = rb.fit_transform(df['Amount'].values.reshape(-1,1))
df.head()

x = df.drop('Class', axis=1)
y = df['Class']

# train and test split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

# spot check algorithms
classifiers = {"Logistic Regression": LogisticRegression(),
               "DecisionTree": DecisionTreeClassifier(),
               "LDA": LinearDiscriminantAnalysis()}
# as the dataset is too big computation time will be high
# bcoz of which iam using only 3 classifiers

for name, clf in classifiers.items():
    accuracy = cross_val_score(clf, xTrain, yTrain, scoring='accuracy', cv=5)
    accuracyTest = cross_val_score(clf, xTest, yTest, scoring='accuracy', cv=5)

    precision = cross_val_score(clf, xTrain, yTrain, scoring='precision', cv=5)
    precisionTest = cross_val_score(clf, xTest, yTest, scoring='precision', cv=5)

    recall = cross_val_score(clf, xTrain, yTrain, scoring='recall', cv=5)
    recallTest = cross_val_score(clf, xTest, yTest, scoring='recall', cv=5)

    print(name, '---', 'Train-Accuracy :%0.2f%%' % (accuracy.mean() * 100),
          'Train-Precision: %0.2f%%' % (precision.mean() * 100),
          'Train-Recall   : %0.2f%%' % (recall.mean() * 100))

    print(name, '---', 'Test-Accuracy :%0.2f%%' % (accuracyTest.mean() * 100),
          'Test-Precision: %0.2f%%' % (precisionTest.mean() * 100),
          'Test-Recall   : %0.2f%%' % (recallTest.mean() * 100), '\n')

#step 1
xTrain_rus,xTest_rus,yTrain_rus,yTest_rus = train_test_split(xTrain,yTrain,test_size = 0.2,random_state = 42)

#step 2
rus = RandomUnderSampler()
x_rus,y_rus = rus.fit_sample(xTrain_rus,yTrain_rus)

#converting it to DataFrame to Visualize in pandas
df_x_rus = pd.DataFrame(x_rus)
df_x_rus['target'] = y_rus
print(df_x_rus['target'].value_counts())
print(df_x_rus['target'].value_counts().plot(kind = 'bar',title = 'RandomUnderSampling\nFrauds = 1 | NoFrauds = 0'))

plt.show()

lr = LogisticRegression()
lr.fit(x_rus,y_rus)

#step 4
yPred_rus = lr.predict(xTest_rus)

rus_accuracy = accuracy_score(yTest_rus,yPred_rus)
rus_classReport = classification_report(yTest_rus,yPred_rus)
#print('\nTrain-Accuracy %0.2f%%'%(rus_accuracy*100),
#      '\nTrain-ClassificationReport:\n',rus_classReport,'\n')

#step 5
yPred_actual = lr.predict(xTest)
test_accuracy = accuracy_score(yTest,yPred_actual)
test_classReport = classification_report(yTest,yPred_actual)
print('\nTest-Accuracy %0.2f%%'%(test_accuracy*100),
      '\n\nTest-ClassificationReport:\n',test_classReport)

plt.show()


#step 1
xTrain_ros,xTest_ros,yTrain_ros,yTest_ros = train_test_split(xTrain,yTrain,test_size=0.2,random_state=42)

#step 2
ros = RandomOverSampler()
x_ros,y_ros = ros.fit_sample(xTrain_ros,yTrain_ros)

#Converting it to dataframe to visualize in pandas
df_x_ros = pd.DataFrame(x_ros)
df_x_ros['target'] = y_ros
print(df_x_ros['target'].value_counts())
print(df_x_ros['target'].value_counts().plot(kind = 'bar',title = 'RandomOverSampling\nFrauds = 0 | NoFrauds = 1'))
plt.show()

#step 3
lr = LogisticRegression()
lr.fit(x_ros,y_ros)

#step 4
yPred_ros = lr.predict(xTest_ros)

ros_accuracy = accuracy_score(yTest_ros,yPred_ros)
ros_classReport = classification_report(yTest_ros,yPred_ros)
print('\nTrain-Accuracy %0.2f%%'%(rus_accuracy*100),
      '\nTrain-ClassificationReport:\n',rus_classReport,'\n')

#step 5
yPred_actual = lr.predict(xTest)
test_accuracy = accuracy_score(yTest,yPred_actual)
test_classReport = classification_report(yTest,yPred_actual)
print('\nTest-Accuracy %0.2f%%'%(test_accuracy*100),
      '\n\nTest-ClassificationReport:\n',test_classReport)


#step 1
xTrain_smote,xTest_smote,yTrain_smote,yTest_smote = train_test_split(xTrain,yTrain,test_size = 0.2,random_state = 42 )

#step2
smote = SMOTE()
x_smote,y_smote = smote.fit_sample(xTrain_smote,yTrain_smote)
#Converting it to dataframe to visualize in pandas
df_x_smote = pd.DataFrame(x_smote)
df_x_smote['target'] = y_smote
print(df_x_smote['target'].value_counts())
print(df_x_smote['target'].value_counts().plot(kind = 'bar',title = 'SMOTE\nFrauds = 0 | NoFrauds = 1'))


rfc = RandomForestClassifier(random_state = 42)
rfc.fit(x_smote,y_smote)
ypred_smote = rfc.predict(xTest_smote)

rfc_prediction=rfc.predict(xTest)
print('RFC-Accuracy',accuracy_score(yTest,rfc_prediction),'\n')
print('Confusion_Matrix:\n',confusion_matrix(yTest,rfc_prediction),'\n')
print('Classification Report',classification_report(yTest,rfc_prediction))


#auc score
rfc_fpr,rfc_tpr,_ = roc_curve(yTest,rfc_prediction)
rfc_auc = auc(rfc_fpr,rfc_tpr)
print('RandomForestClassifier-auc : %0.2f%%'%(rfc_auc * 100))

#roc curve
plt.figure()
plt.plot(rfc_fpr,rfc_tpr,label ='RFC(auc = %0.2f%%)'%(rfc_auc *100))
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title('Smote with RandomForestClassifier\nROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

