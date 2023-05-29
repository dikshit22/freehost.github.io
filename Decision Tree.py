#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pickle


# In[27]:


#Reading Data
data = pd.read_csv('./DASS21Dataset.csv')


# # Data Descrption
# Depression, Anxiety and Stress Questionairs

# In[28]:


#Viewing dataframe
data


# In[30]:


#Displaying all info of the data
data.info()


# In[31]:


#Checking for NA values
print('Display NA values in each column')
data.isna().sum()


# In[32]:


#Checking for NULL values
print('Display NULL values in each column')
data.isnull().sum()


# In[33]:


#Checking the datatypes
data.dtypes


# In[34]:


#Splitting the dataframe
D_data = data.loc[:,[(data[col] == "(d)").any() for col in data.columns]]
A_data = data.loc[:,[(data[col] == "(a)").any() for col in data.columns]]
S_data = data.loc[:,[(data[col] == "(s)").any() for col in data.columns]]


# In[35]:


#Dropping unwanted rows
D_data.drop([0, 1], axis=0, inplace=True)
A_data.drop([0, 1], axis=0, inplace=True)
S_data.drop([0, 1], axis=0, inplace=True)


# In[36]:


D_data


# In[37]:


A_data


# In[38]:


S_data


# In[39]:


#Changing the datatypes
D_data = D_data.astype(int)
A_data = A_data.astype(int)
S_data = S_data.astype(int)


# In[40]:


D_data.dtypes


# In[41]:


A_data.dtypes


# In[42]:


S_data.dtypes


# In[43]:


#Reassigning the values to the target column
D_data.loc[D_data['Depression'] < 10, 'Depression'] = 0
D_data.loc[(D_data['Depression'] >= 10) & (D_data['Depression'] <= 13), 'Depression'] = 1
D_data.loc[(D_data['Depression'] >= 14) & (D_data['Depression'] <= 20), 'Depression'] = 2
D_data.loc[(D_data['Depression'] >= 21) & (D_data['Depression'] <= 27), 'Depression'] = 3
D_data.loc[D_data['Depression'] >= 28, 'Depression'] = 4

A_data.loc[A_data['Anxiety'] < 8, 'Anxiety'] = 0
A_data.loc[(A_data['Anxiety'] >= 8) & (A_data['Anxiety'] <= 9), 'Anxiety'] = 1
A_data.loc[(A_data['Anxiety'] >= 10) & (A_data['Anxiety'] <= 14), 'Anxiety'] = 2
A_data.loc[(A_data['Anxiety'] >= 15) & (A_data['Anxiety'] <= 19), 'Anxiety'] = 3
A_data.loc[A_data['Anxiety'] >= 20, 'Anxiety'] = 4

S_data.loc[S_data['Stress'] < 15, 'Stress'] = 0
S_data.loc[(S_data['Stress'] >= 15) & (S_data['Stress'] <= 18), 'Stress'] = 1
S_data.loc[(S_data['Stress'] >= 19) & (S_data['Stress'] <= 25), 'Stress'] = 2
S_data.loc[(S_data['Stress'] >= 26) & (S_data['Stress'] <= 33), 'Stress'] = 3
S_data.loc[S_data['Stress'] >= 34, 'Stress'] = 4


# In[44]:


D_data


# In[45]:


A_data


# In[46]:


S_data


# In[47]:


#Printing number of each unique values
print("Depression:")
print('0 - ', (D_data['Depression'] == 0).sum())
print('1 - ', (D_data['Depression'] == 1).sum())
print('2 - ', (D_data['Depression'] == 2).sum())
print('3 - ', (D_data['Depression'] == 3).sum())
print('4 - ', (D_data['Depression'] == 4).sum())

print("\nAnxiety:")
print('0 - ', (A_data['Anxiety'] == 0).sum())
print('1 - ', (A_data['Anxiety'] == 1).sum())
print('2 - ', (A_data['Anxiety'] == 2).sum())
print('3 - ', (A_data['Anxiety'] == 3).sum())
print('4 - ', (A_data['Anxiety'] == 4).sum())

print("\nStress:")
print('0 - ', (S_data['Stress'] == 0).sum())
print('1 - ', (S_data['Stress'] == 1).sum())
print('2 - ', (S_data['Stress'] == 2).sum())
print('3 - ', (S_data['Stress'] == 3).sum())
print('4 - ', (S_data['Stress'] == 4).sum())


# # ----------Decision Tree----------

# In[48]:


#Separating features (X) and target variable (y)
D_X = D_data.drop('Depression', axis=1)
D_y = D_data['Depression']
#Splitting the data into training and testing sets
D_X_train, D_X_test, D_y_train, D_y_test = train_test_split(D_X, D_y, test_size=0.25, random_state=6)
#Creating the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
D_clf = DecisionTreeClassifier()


# In[49]:


#Training the model
D_clf.fit(D_X_train, D_y_train)
#Testing the model
D_y_pred = D_clf.predict(D_X_test)


# In[50]:


# Viewing the classification report
print(classification_report(D_y_test, D_y_pred))


# In[57]:


# Viewing the confusion matrix
D_cm = confusion_matrix(D_y_test, D_y_pred)
print("Confusion Matrix\n", D_cm)


# In[58]:


#Calculating different outcomes
sum_D_cm = np.sum(D_cm)
print('Classes  TP \t FN \t FP \t TN')
D_tp_0 = D_cm[0, 0]
D_fn_0 = sum(D_cm[0]) - D_tp_0
D_fp_0 = sum(D_cm[:, 0]) - D_tp_0
D_tn_0 = sum_D_cm - (D_tp_0 + D_fn_0 + D_fp_0)
print('0\t', D_tp_0, '\t', D_fn_0, '\t', D_fp_0, '\t', D_tn_0)
D_tp_1 = D_cm[1, 1]
D_fn_1 = sum(D_cm[1]) - D_tp_1
D_fp_1 = sum(D_cm[:, 1]) - D_tp_1
D_tn_1 = sum_D_cm - (D_tp_1 + D_fn_1 + D_fp_1)
print('1\t', D_tp_1, '\t', D_fn_1, '\t', D_fp_1, '\t', D_tn_1)
D_tp_2 = D_cm[2, 2]
D_fn_2 = sum(D_cm[2]) - D_tp_2
D_fp_2 = sum(D_cm[:, 2]) - D_tp_2
D_tn_2 = sum_D_cm - (D_tp_2 + D_fn_2 + D_fp_2)
print('2\t', D_tp_2, '\t', D_fn_2, '\t', D_fp_2, '\t', D_tn_2)
D_tp_3 = D_cm[3, 3]
D_fn_3 = sum(D_cm[3]) - D_tp_3
D_fp_3 = sum(D_cm[:, 3]) - D_tp_3
D_tn_3 = sum_D_cm - (D_tp_3 + D_fn_3 + D_fp_3)
print('3\t', D_tp_3, '\t', D_fn_3, '\t', D_fp_3, '\t', D_tn_3)
D_tp_4 = D_cm[4, 4]
D_fn_4 = sum(D_cm[4]) - D_tp_4
D_fp_4 = sum(D_cm[:, 4]) - D_tp_4
D_tn_4 = sum_D_cm - (D_tp_4 + D_fn_4 + D_fp_4)
print('4\t', D_tp_4, '\t', D_fn_4, '\t', D_fp_4, '\t', D_tn_4)

D_tp = D_tp_0 + D_tp_1 + D_tp_2 + D_tp_3 + D_tp_4
D_fn = D_fn_0 + D_fn_1 + D_fn_2 + D_fn_3 + D_fn_4
D_fp = D_fp_0 + D_fp_1 + D_fp_2 + D_fp_3 + D_fp_4
D_tn = D_tn_0 + D_tn_1 + D_tn_2 + D_tn_3 + D_tn_4
print('\nTotal\t', D_tp, '\t', D_fn, '\t', D_fp, '\t', D_tn)


# In[59]:


#Calculating the performance on different parameters
D_accuracy = accuracy_score(D_y_test, D_y_pred)
print("Accuracy is ", round(D_accuracy, 2))
D_precision = precision_score(D_y_test, D_y_pred, average='weighted')
print("Precision is ", round(D_precision, 2))
D_recall = recall_score(D_y_test, D_y_pred, average='weighted')
print("Recall is ", round(D_recall, 2))
D_specificity = D_tn / (D_tn + D_fp)
print("Specificity is ", round(D_specificity, 2))
D_f1 = f1_score(D_y_test, D_y_pred, average='weighted')
print("F1 is ", round(D_f1, 2))


# In[60]:


#Separating features (X) and target variable (y)
A_y = A_data['Anxiety']
A_X = A_data.drop('Anxiety', axis=1)
#Splitting the data into training and testing test
A_X_train, A_X_test, A_y_train, A_y_test = train_test_split(A_X, A_y, test_size=0.25, random_state=6)
#Creating the Decision Tree model
A_clf = DecisionTreeClassifier()


# In[61]:


#Training the model
A_clf.fit(A_X_train, A_y_train)
#Testing the model
A_y_pred = A_clf.predict(A_X_test)


# In[62]:


#Viewing the classification report
print(classification_report(A_y_test, A_y_pred))


# In[63]:


#Viewing the confusion matrix
A_cm = confusion_matrix(A_y_test, A_y_pred)
print("Confusion Matrix\n", A_cm)


# In[64]:


#Calculating different outcomes
sum_A_cm = np.sum(A_cm)
print('Classes  TP \t FN \t FP \t TN')
A_tp_0 = A_cm[0, 0]
A_fn_0 = sum(A_cm[0]) - A_tp_0
A_fp_0 = sum(A_cm[:, 0]) - A_tp_0
A_tn_0 = sum_A_cm - (A_tp_0 + A_fn_0 + A_fp_0)
print('0\t', A_tp_0, '\t', A_fn_0, '\t', A_fp_0, '\t', A_tn_0)
A_tp_1 = A_cm[1, 1]
A_fn_1 = sum(A_cm[1]) - A_tp_1
A_fp_1 = sum(A_cm[:, 1]) - A_tp_1
A_tn_1 = sum_A_cm - (A_tp_1 + A_fn_1 + A_fp_1)
print('1\t', A_tp_1, '\t', A_fn_1, '\t', A_fp_1, '\t', A_tn_1)
A_tp_2 = A_cm[2, 2]
A_fn_2 = sum(A_cm[2]) - A_tp_2
A_fp_2 = sum(A_cm[:, 2]) - A_tp_2
A_tn_2 = sum_A_cm - (A_tp_2 + A_fn_2 + A_fp_2)
print('2\t', A_tp_2, '\t', A_fn_2, '\t', A_fp_2, '\t', A_tn_2)
A_tp_3 = A_cm[3, 3]
A_fn_3 = sum(A_cm[3]) - A_tp_3
A_fp_3 = sum(A_cm[:, 3]) - A_tp_3
A_tn_3 = sum_A_cm - (A_tp_3 + A_fn_3 + A_fp_3)
print('3\t', A_tp_3, '\t', A_fn_3, '\t', A_fp_3, '\t', A_tn_3)
A_tp_4 = A_cm[4, 4]
A_fn_4 = sum(A_cm[4]) - A_tp_4
A_fp_4 = sum(A_cm[:, 4]) - A_tp_4
A_tn_4 = sum_A_cm - (A_tp_4 + A_fn_4 + A_fp_4)
print('4\t', A_tp_4, '\t', A_fn_4, '\t', A_fp_4, '\t', A_tn_4)

A_tp = A_tp_0 + A_tp_1 + A_tp_2 + A_tp_3 + A_tp_4
A_fn = A_fn_0 + A_fn_1 + A_fn_2 + A_fn_3 + A_fn_4
A_fp = A_fp_0 + A_fp_1 + A_fp_2 + A_fp_3 + A_fp_4
A_tn = A_tn_0 + A_tn_1 + A_tn_2 + A_tn_3 + A_tn_4
print('\nTotal\t', A_tp, '\t', A_fn, '\t', A_fp, '\t', A_tn)


# In[65]:


#Calculating the performance on different parameters
A_accuracy = accuracy_score(A_y_test, A_y_pred)
print("Accuracy is ", round(A_accuracy, 2))
A_precision = precision_score(A_y_test, A_y_pred, average='weighted')
print("Precision is ", round(A_precision, 2))
A_recall = recall_score(A_y_test, A_y_pred, average='weighted')
print("Recall is ", round(A_recall, 2))
A_specificity = A_tn / (A_tn + A_fp)
print("Specificity is ", round(A_specificity, 2))
A_f1 = f1_score(A_y_test, A_y_pred, average='weighted')
print("F1 is ", round(A_f1, 2))


# In[66]:


#Separating features (X) and target variable (y)
S_y = S_data['Stress']
S_X = S_data.drop('Stress', axis=1)
#Splitting the data into training and testing test
S_X_train, S_X_test, S_y_train, S_y_test = train_test_split(S_X, S_y, test_size=0.25, random_state=6)
#Creating the Decision Tree model
S_clf = DecisionTreeClassifier()


# In[67]:


#Training the model
S_clf.fit(S_X_train, S_y_train)
#Testing the model
S_y_pred = S_clf.predict(S_X_test)


# In[68]:


#Viewing the classification report
print(classification_report(S_y_test, S_y_pred))


# In[69]:


#Viewing the confusion matrix
S_cm = confusion_matrix(S_y_test, S_y_pred)
print("Confusion Matrix\n", S_cm)


# In[70]:


#Calculating different outcomes
sum_S_cm = np.sum(S_cm)
print('Classes  TP \t FN \t FP \t TN')
S_tp_0 = S_cm[0, 0]
S_fn_0 = sum(S_cm[0]) - S_tp_0
S_fp_0 = sum(S_cm[:, 0]) - S_tp_0
S_tn_0 = sum_S_cm - (S_tp_0 + S_fn_0 + S_fp_0)
print('0\t', S_tp_0, '\t', S_fn_0, '\t', S_fp_0, '\t', S_tn_0)
S_tp_1 = S_cm[1, 1]
S_fn_1 = sum(S_cm[1]) - S_tp_1
S_fp_1 = sum(S_cm[:, 1]) - S_tp_1
S_tn_1 = sum_S_cm - (S_tp_1 + S_fn_1 + S_fp_1)
print('1\t', S_tp_1, '\t', S_fn_1, '\t', S_fp_1, '\t', S_tn_1)
S_tp_2 = S_cm[2, 2]
S_fn_2 = sum(S_cm[2]) - S_tp_2
S_fp_2 = sum(S_cm[:, 2]) - S_tp_2
S_tn_2 = sum_S_cm - (S_tp_2 + S_fn_2 + S_fp_2)
print('2\t', S_tp_2, '\t', S_fn_2, '\t', S_fp_2, '\t', S_tn_2)
S_tp_3 = S_cm[3, 3]
S_fn_3 = sum(S_cm[3]) - S_tp_3
S_fp_3 = sum(S_cm[:, 3]) - S_tp_3
S_tn_3 = sum_S_cm - (S_tp_3 + S_fn_3 + S_fp_3)
print('3\t', S_tp_3, '\t', S_fn_3, '\t', S_fp_3, '\t', S_tn_3)
S_tp_4 = S_cm[4, 4]
S_fn_4 = sum(S_cm[4]) - S_tp_4
S_fp_4 = sum(S_cm[:, 4]) - S_tp_4
S_tn_4 = sum_S_cm - (S_tp_4 + S_fn_4 + S_fp_4)
print('4\t', S_tp_4, '\t', S_fn_4, '\t', S_fp_4, '\t', S_tn_4)

S_tp = S_tp_0 + S_tp_1 + S_tp_2 + S_tp_3 + S_tp_4
S_fn = S_fn_0 + S_fn_1 + S_fn_2 + S_fn_3 + S_fn_4
S_fp = S_fp_0 + S_fp_1 + S_fp_2 + S_fp_3 + S_fp_4
S_tn = S_tn_0 + S_tn_1 + S_tn_2 + S_tn_3 + S_tn_4
print('\nTotal\t', S_tp, '\t', S_fn, '\t', S_fp, '\t', S_tn)


# In[71]:


#Calculating the performance on different parameters
S_accuracy = accuracy_score(S_y_test, S_y_pred)
print("Accuracy is ", round(S_accuracy, 2))
S_precision = precision_score(S_y_test, S_y_pred, average='weighted')
print("Precision is ", round(S_precision, 2))
S_recall = recall_score(S_y_test, S_y_pred, average='weighted')
print("Recall is ", round(S_recall, 2))
S_specificity = S_tn / (S_tn + S_fp)
print("Specificity is ", round(S_specificity, 2))
S_f1 = f1_score(S_y_test, S_y_pred, average='weighted')
print("F1 is ", round(S_f1, 2))


# In[72]:


print("\t\tDepression\tAnxiety \tStress")
print("Accuracy\t", round(D_accuracy, 2), "\t\t", round(A_accuracy, 2), "\t\t", round(S_accuracy, 2))
print("Precision\t", round(D_precision, 2), "\t\t", round(A_precision, 2), "\t\t", round(S_precision, 2))
print("Recall\t\t", round(D_recall, 2), "\t\t", round(A_recall, 2), "\t\t", round(S_recall, 2))
print("Specificity\t", round(D_specificity, 2), "\t\t", round(A_specificity, 2), "\t\t", round(S_specificity, 2))
print("F1\t\t", round(D_f1, 2), "\t\t", round(A_f1, 2), "\t\t", round(S_f1, 2))


# # ----------Random Forest----------

# In[73]:


#Creating the Random Forest
from sklearn.ensemble import RandomForestClassifier
D_rf = RandomForestClassifier()


# In[74]:


#Training the model
D_rf.fit(D_X_train, D_y_train)
#Testing the model
D_y_pred = D_rf.predict(D_X_test)


# In[75]:


#Viewing the classification report
print(classification_report(D_y_test, D_y_pred))


# In[76]:


#Viewing the confusion matirx
D_cm = confusion_matrix(D_y_test, D_y_pred)
print(D_cm)


# In[77]:


#Calculating different outcomes
sum_D_cm = np.sum(D_cm)
print('Classes  TP \t FN \t FP \t TN')
D_tp_0 = D_cm[0, 0]
D_fn_0 = sum(D_cm[0]) - D_tp_0
D_fp_0 = sum(D_cm[:, 0]) - D_tp_0
D_tn_0 = sum_D_cm - (D_tp_0 + D_fn_0 + D_fp_0)
print('0\t', D_tp_0, '\t', D_fn_0, '\t', D_fp_0, '\t', D_tn_0)
D_tp_1 = D_cm[1, 1]
D_fn_1 = sum(D_cm[1]) - D_tp_1
D_fp_1 = sum(D_cm[:, 1]) - D_tp_1
D_tn_1 = sum_D_cm - (D_tp_1 + D_fn_1 + D_fp_1)
print('1\t', D_tp_1, '\t', D_fn_1, '\t', D_fp_1, '\t', D_tn_1)
D_tp_2 = D_cm[2, 2]
D_fn_2 = sum(D_cm[2]) - D_tp_2
D_fp_2 = sum(D_cm[:, 2]) - D_tp_2
D_tn_2 = sum_D_cm - (D_tp_2 + D_fn_2 + D_fp_2)
print('2\t', D_tp_2, '\t', D_fn_2, '\t', D_fp_2, '\t', D_tn_2)
D_tp_3 = D_cm[3, 3]
D_fn_3 = sum(D_cm[3]) - D_tp_3
D_fp_3 = sum(D_cm[:, 3]) - D_tp_3
D_tn_3 = sum_D_cm - (D_tp_3 + D_fn_3 + D_fp_3)
print('3\t', D_tp_3, '\t', D_fn_3, '\t', D_fp_3, '\t', D_tn_3)
D_tp_4 = D_cm[4, 4]
D_fn_4 = sum(D_cm[4]) - D_tp_4
D_fp_4 = sum(D_cm[:, 4]) - D_tp_4
D_tn_4 = sum_D_cm - (D_tp_4 + D_fn_4 + D_fp_4)
print('4\t', D_tp_4, '\t', D_fn_4, '\t', D_fp_4, '\t', D_tn_4)

D_tp = D_tp_0 + D_tp_1 + D_tp_2 + D_tp_3 + D_tp_4
D_fn = D_fn_0 + D_fn_1 + D_fn_2 + D_fn_3 + D_fn_4
D_fp = D_fp_0 + D_fp_1 + D_fp_2 + D_fp_3 + D_fp_4
D_tn = D_tn_0 + D_tn_1 + D_tn_2 + D_tn_3 + D_tn_4
print('\nTotal\t', D_tp, '\t', D_fn, '\t', D_fp, '\t', D_tn)


# In[78]:


#Calculating the performance on different parameters
D_accuracy = accuracy_score(D_y_test, D_y_pred)
print("Accuracy is ", round(D_accuracy, 2))
D_precision = precision_score(D_y_test, D_y_pred, average='weighted')
print("Precision is ", round(D_precision, 2))
D_recall = recall_score(D_y_test, D_y_pred, average='weighted')
print("Recall is ", round(D_recall, 2))
D_specificity = D_tn / (D_tn + D_fp)
print("Specificity is ", round(D_specificity, 2))
D_f1 = f1_score(D_y_test, D_y_pred, average='weighted')
print("F1 is ", round(D_f1, 2))


# In[79]:


#Creating the Random Forest model
A_rf = RandomForestClassifier()


# In[167]:


#Training the model
A_rf.fit(A_X_train, A_y_train)
#Testing the model
A_y_pred = A_rf.predict(A_X_test)


# In[168]:


#Viewing the classification report
print(classification_report(A_y_test, A_y_pred))


# In[82]:


#Viewing the confusion matrix
A_cm = confusion_matrix(A_y_test, A_y_pred)
print("Confusion Matrix\n", A_cm)


# In[83]:


#Calculating different outcomes
sum_A_cm = np.sum(A_cm)
print('Classes  TP \t FN \t FP \t TN')
A_tp_0 = A_cm[0, 0]
A_fn_0 = sum(A_cm[0]) - A_tp_0
A_fp_0 = sum(A_cm[:, 0]) - A_tp_0
A_tn_0 = sum_A_cm - (A_tp_0 + A_fn_0 + A_fp_0)
print('0\t', A_tp_0, '\t', A_fn_0, '\t', A_fp_0, '\t', A_tn_0)
A_tp_1 = A_cm[1, 1]
A_fn_1 = sum(A_cm[1]) - A_tp_1
A_fp_1 = sum(A_cm[:, 1]) - A_tp_1
A_tn_1 = sum_A_cm - (A_tp_1 + A_fn_1 + A_fp_1)
print('1\t', A_tp_1, '\t', A_fn_1, '\t', A_fp_1, '\t', A_tn_1)
A_tp_2 = A_cm[2, 2]
A_fn_2 = sum(A_cm[2]) - A_tp_2
A_fp_2 = sum(A_cm[:, 2]) - A_tp_2
A_tn_2 = sum_A_cm - (A_tp_2 + A_fn_2 + A_fp_2)
print('2\t', A_tp_2, '\t', A_fn_2, '\t', A_fp_2, '\t', A_tn_2)
A_tp_3 = A_cm[3, 3]
A_fn_3 = sum(A_cm[3]) - A_tp_3
A_fp_3 = sum(A_cm[:, 3]) - A_tp_3
A_tn_3 = sum_A_cm - (A_tp_3 + A_fn_3 + A_fp_3)
print('3\t', A_tp_3, '\t', A_fn_3, '\t', A_fp_3, '\t', A_tn_3)
A_tp_4 = A_cm[4, 4]
A_fn_4 = sum(A_cm[4]) - A_tp_4
A_fp_4 = sum(A_cm[:, 4]) - A_tp_4
A_tn_4 = sum_A_cm - (A_tp_4 + A_fn_4 + A_fp_4)
print('4\t', A_tp_4, '\t', A_fn_4, '\t', A_fp_4, '\t', A_tn_4)

A_tp = A_tp_0 + A_tp_1 + A_tp_2 + A_tp_3 + A_tp_4
A_fn = A_fn_0 + A_fn_1 + A_fn_2 + A_fn_3 + A_fn_4
A_fp = A_fp_0 + A_fp_1 + A_fp_2 + A_fp_3 + A_fp_4
A_tn = A_tn_0 + A_tn_1 + A_tn_2 + A_tn_3 + A_tn_4
print('\nTotal\t', A_tp, '\t', A_fn, '\t', A_fp, '\t', A_tn)


# In[84]:


#Calculating the performance on different parameters
A_accuracy = accuracy_score(A_y_test, A_y_pred)
print("Accuracy is ", round(A_accuracy, 2))
A_precision = precision_score(A_y_test, A_y_pred, average='weighted')
print("Precision is ", round(A_precision, 2))
A_recall = recall_score(A_y_test, A_y_pred, average='weighted')
print("Recall is ", round(A_recall, 2))
A_specificity = A_tn / (A_tn + A_fp)
print("Specificity is ", round(A_specificity, 2))
A_f1 = f1_score(A_y_test, A_y_pred, average='weighted')
print("F1 is ", round(A_f1, 2))


# In[85]:


#Creating the Random Forest model
S_rf = RandomForestClassifier()


# In[86]:


#Training the model
S_rf.fit(S_X_train, S_y_train)
#Testing the model
S_y_pred = S_rf.predict(S_X_test)


# In[87]:


#Viewing the classification report
print(classification_report(S_y_test, S_y_pred))


# In[88]:


#Viewing the confusion matrix
S_cm = confusion_matrix(S_y_test, S_y_pred)
print("Confusion Matrix\n", S_cm)


# In[89]:


#Calculating different outcomes
sum_S_cm = np.sum(S_cm)
print('Classes  TP \t FN \t FP \t TN')
S_tp_0 = S_cm[0, 0]
S_fn_0 = sum(S_cm[0]) - S_tp_0
S_fp_0 = sum(S_cm[:, 0]) - S_tp_0
S_tn_0 = sum_S_cm - (S_tp_0 + S_fn_0 + S_fp_0)
print('0\t', S_tp_0, '\t', S_fn_0, '\t', S_fp_0, '\t', S_tn_0)
S_tp_1 = S_cm[1, 1]
S_fn_1 = sum(S_cm[1]) - S_tp_1
S_fp_1 = sum(S_cm[:, 1]) - S_tp_1
S_tn_1 = sum_S_cm - (S_tp_1 + S_fn_1 + S_fp_1)
print('1\t', S_tp_1, '\t', S_fn_1, '\t', S_fp_1, '\t', S_tn_1)
S_tp_2 = S_cm[2, 2]
S_fn_2 = sum(S_cm[2]) - S_tp_2
S_fp_2 = sum(S_cm[:, 2]) - S_tp_2
S_tn_2 = sum_S_cm - (S_tp_2 + S_fn_2 + S_fp_2)
print('2\t', S_tp_2, '\t', S_fn_2, '\t', S_fp_2, '\t', S_tn_2)
S_tp_3 = S_cm[3, 3]
S_fn_3 = sum(S_cm[3]) - S_tp_3
S_fp_3 = sum(S_cm[:, 3]) - S_tp_3
S_tn_3 = sum_S_cm - (S_tp_3 + S_fn_3 + S_fp_3)
print('3\t', S_tp_3, '\t', S_fn_3, '\t', S_fp_3, '\t', S_tn_3)
S_tp_4 = S_cm[4, 4]
S_fn_4 = sum(S_cm[4]) - S_tp_4
S_fp_4 = sum(S_cm[:, 4]) - S_tp_4
S_tn_4 = sum_S_cm - (S_tp_4 + S_fn_4 + S_fp_4)
print('4\t', S_tp_4, '\t', S_fn_4, '\t', S_fp_4, '\t', S_tn_4)

S_tp = S_tp_0 + S_tp_1 + S_tp_2 + S_tp_3 + S_tp_4
S_fn = S_fn_0 + S_fn_1 + S_fn_2 + S_fn_3 + S_fn_4
S_fp = S_fp_0 + S_fp_1 + S_fp_2 + S_fp_3 + S_fp_4
S_tn = S_tn_0 + S_tn_1 + S_tn_2 + S_tn_3 + S_tn_4
print('\nTotal\t', S_tp, '\t', S_fn, '\t', S_fp, '\t', S_tn)


# In[90]:


#Calculating the performance on different parameters
S_accuracy = accuracy_score(S_y_test, S_y_pred)
print("Accuracy is ", round(S_accuracy, 2))
S_precision = precision_score(S_y_test, S_y_pred, average='weighted')
print("Precision is ", round(S_precision, 2))
S_recall = recall_score(S_y_test, S_y_pred, average='weighted')
print("Recall is ", round(S_recall, 2))
S_specificity = S_tn / (S_tn + S_fp)
print("Specificity is ", round(S_specificity, 2))
S_f1 = f1_score(S_y_test, S_y_pred, average='weighted')
print("F1 is ", round(S_f1, 2))


# In[91]:


print("\t\tDepression\tAnxiety \tStress")
print("Accuracy\t", round(D_accuracy, 2), "\t\t", round(A_accuracy, 2), "\t\t", round(S_accuracy, 2))
print("Precision\t", round(D_precision, 2), "\t\t", round(A_precision, 2), "\t\t", round(S_precision, 2))
print("Recall\t\t", round(D_recall, 2), "\t\t", round(A_recall, 2), "\t\t", round(S_recall, 2))
print("Specificity\t", round(D_specificity, 2), "\t\t", round(A_specificity, 2), "\t\t", round(S_specificity, 2))
print("F1\t\t", round(D_f1, 2), "\t\t", round(A_f1, 2), "\t\t", round(S_f1, 2))


# # ----------SVM(Support Vector Machine)----------

# In[108]:


#Creating the SVM model
from sklearn import svm
D_svm = svm.SVC(C=1, kernel='linear')


# In[109]:


#Training the model
for i in range(2, 603, 100):
    D_svm.fit(D_X_train[i:i+100], D_y_train[i:i+100])
#Testing the model
D_y_pred = D_svm.predict(D_X_test)


# In[110]:


#Viewing the classification report
print(classification_report(D_y_test, D_y_pred))


# In[111]:


#Viewing the confusion matirx
D_cm = confusion_matrix(D_y_test, D_y_pred)
print(D_cm)


# In[112]:


#Calculating different outcomes
sum_D_cm = np.sum(D_cm)
print('Classes  TP \t FN \t FP \t TN')
D_tp_0 = D_cm[0, 0]
D_fn_0 = sum(D_cm[0]) - D_tp_0
D_fp_0 = sum(D_cm[:, 0]) - D_tp_0
D_tn_0 = sum_D_cm - (D_tp_0 + D_fn_0 + D_fp_0)
print('0\t', D_tp_0, '\t', D_fn_0, '\t', D_fp_0, '\t', D_tn_0)
D_tp_1 = D_cm[1, 1]
D_fn_1 = sum(D_cm[1]) - D_tp_1
D_fp_1 = sum(D_cm[:, 1]) - D_tp_1
D_tn_1 = sum_D_cm - (D_tp_1 + D_fn_1 + D_fp_1)
print('1\t', D_tp_1, '\t', D_fn_1, '\t', D_fp_1, '\t', D_tn_1)
D_tp_2 = D_cm[2, 2]
D_fn_2 = sum(D_cm[2]) - D_tp_2
D_fp_2 = sum(D_cm[:, 2]) - D_tp_2
D_tn_2 = sum_D_cm - (D_tp_2 + D_fn_2 + D_fp_2)
print('2\t', D_tp_2, '\t', D_fn_2, '\t', D_fp_2, '\t', D_tn_2)
D_tp_3 = D_cm[3, 3]
D_fn_3 = sum(D_cm[3]) - D_tp_3
D_fp_3 = sum(D_cm[:, 3]) - D_tp_3
D_tn_3 = sum_D_cm - (D_tp_3 + D_fn_3 + D_fp_3)
print('3\t', D_tp_3, '\t', D_fn_3, '\t', D_fp_3, '\t', D_tn_3)
D_tp_4 = D_cm[4, 4]
D_fn_4 = sum(D_cm[4]) - D_tp_4
D_fp_4 = sum(D_cm[:, 4]) - D_tp_4
D_tn_4 = sum_D_cm - (D_tp_4 + D_fn_4 + D_fp_4)
print('4\t', D_tp_4, '\t', D_fn_4, '\t', D_fp_4, '\t', D_tn_4)

D_tp = D_tp_0 + D_tp_1 + D_tp_2 + D_tp_3 + D_tp_4
D_fn = D_fn_0 + D_fn_1 + D_fn_2 + D_fn_3 + D_fn_4
D_fp = D_fp_0 + D_fp_1 + D_fp_2 + D_fp_3 + D_fp_4
D_tn = D_tn_0 + D_tn_1 + D_tn_2 + D_tn_3 + D_tn_4
print('\nTotal\t', D_tp, '\t', D_fn, '\t', D_fp, '\t', D_tn)


# In[113]:


#Calculating the performance on different parameters
D_accuracy = accuracy_score(D_y_test, D_y_pred)
print("Accuracy is ", round(D_accuracy, 2))
D_precision = precision_score(D_y_test, D_y_pred, average='weighted')
print("Precision is ", round(D_precision, 2))
D_recall = recall_score(D_y_test, D_y_pred, average='weighted')
print("Recall is ", round(D_recall, 2))
D_specificity = D_tn / (D_tn + D_fp)
print("Specificity is ", round(D_specificity, 2))
D_f1 = f1_score(D_y_test, D_y_pred, average='weighted')
print("F1 is ", round(D_f1, 2))


# In[117]:


#Creating the SVM model
A_svm = svm.SVC(C=1, kernel='linear')


# In[149]:


#Training the model
for i in range(2, 603, 300):
    A_svm.fit(A_X_train[i:i+100], A_y_train[i:i+100])
#Testing the model
A_y_pred = A_svm.predict(A_X_test)


# In[150]:


#Viewing the classification report
print(classification_report(A_y_test, A_y_pred))


# In[151]:


#Viewing the confusion matrix
A_cm = confusion_matrix(A_y_test, A_y_pred)
print("Confusion Matrix\n", A_cm)


# In[152]:


#Calculating different outcomes
sum_A_cm = np.sum(A_cm)
print('Classes  TP \t FN \t FP \t TN')
A_tp_0 = A_cm[0, 0]
A_fn_0 = sum(A_cm[0]) - A_tp_0
A_fp_0 = sum(A_cm[:, 0]) - A_tp_0
A_tn_0 = sum_A_cm - (A_tp_0 + A_fn_0 + A_fp_0)
print('0\t', A_tp_0, '\t', A_fn_0, '\t', A_fp_0, '\t', A_tn_0)
A_tp_1 = A_cm[1, 1]
A_fn_1 = sum(A_cm[1]) - A_tp_1
A_fp_1 = sum(A_cm[:, 1]) - A_tp_1
A_tn_1 = sum_A_cm - (A_tp_1 + A_fn_1 + A_fp_1)
print('1\t', A_tp_1, '\t', A_fn_1, '\t', A_fp_1, '\t', A_tn_1)
A_tp_2 = A_cm[2, 2]
A_fn_2 = sum(A_cm[2]) - A_tp_2
A_fp_2 = sum(A_cm[:, 2]) - A_tp_2
A_tn_2 = sum_A_cm - (A_tp_2 + A_fn_2 + A_fp_2)
print('2\t', A_tp_2, '\t', A_fn_2, '\t', A_fp_2, '\t', A_tn_2)
A_tp_3 = A_cm[3, 3]
A_fn_3 = sum(A_cm[3]) - A_tp_3
A_fp_3 = sum(A_cm[:, 3]) - A_tp_3
A_tn_3 = sum_A_cm - (A_tp_3 + A_fn_3 + A_fp_3)
print('3\t', A_tp_3, '\t', A_fn_3, '\t', A_fp_3, '\t', A_tn_3)
A_tp_4 = A_cm[4, 4]
A_fn_4 = sum(A_cm[4]) - A_tp_4
A_fp_4 = sum(A_cm[:, 4]) - A_tp_4
A_tn_4 = sum_A_cm - (A_tp_4 + A_fn_4 + A_fp_4)
print('4\t', A_tp_4, '\t', A_fn_4, '\t', A_fp_4, '\t', A_tn_4)

A_tp = A_tp_0 + A_tp_1 + A_tp_2 + A_tp_3 + A_tp_4
A_fn = A_fn_0 + A_fn_1 + A_fn_2 + A_fn_3 + A_fn_4
A_fp = A_fp_0 + A_fp_1 + A_fp_2 + A_fp_3 + A_fp_4
A_tn = A_tn_0 + A_tn_1 + A_tn_2 + A_tn_3 + A_tn_4
print('\nTotal\t', A_tp, '\t', A_fn, '\t', A_fp, '\t', A_tn)


# In[153]:


#Calculating the performance on different parameters
A_accuracy = accuracy_score(A_y_test, A_y_pred)
print("Accuracy is ", round(A_accuracy, 2))
A_precision = precision_score(A_y_test, A_y_pred, average='weighted')
print("Precision is ", round(A_precision, 2))
A_recall = recall_score(A_y_test, A_y_pred, average='weighted')
print("Recall is ", round(A_recall, 2))
A_specificity = A_tn / (A_tn + A_fp)
print("Specificity is ", round(A_specificity, 2))
A_f1 = f1_score(A_y_test, A_y_pred, average='weighted')
print("F1 is ", round(A_f1, 2))


# In[154]:


#Creating the SVM model
S_svm = svm.SVC(C=1, kernel='linear')


# In[155]:


#Training the model
for i in range(2, 603, 100):
    S_svm.fit(S_X_train[i:i+100], S_y_train[i:i+100])
#Testing the model
S_y_pred = S_svm.predict(S_X_test)


# In[156]:


#Viewing the classification report
print(classification_report(S_y_test, S_y_pred))


# In[157]:


#Viewing the confusion matrix
S_cm = confusion_matrix(S_y_test, S_y_pred)
print("Confusion Matrix\n", S_cm)


# In[158]:


#Calculating different outcomes
sum_S_cm = np.sum(S_cm)
print('Classes  TP \t FN \t FP \t TN')
S_tp_0 = S_cm[0, 0]
S_fn_0 = sum(S_cm[0]) - S_tp_0
S_fp_0 = sum(S_cm[:, 0]) - S_tp_0
S_tn_0 = sum_S_cm - (S_tp_0 + S_fn_0 + S_fp_0)
print('0\t', S_tp_0, '\t', S_fn_0, '\t', S_fp_0, '\t', S_tn_0)
S_tp_1 = S_cm[1, 1]
S_fn_1 = sum(S_cm[1]) - S_tp_1
S_fp_1 = sum(S_cm[:, 1]) - S_tp_1
S_tn_1 = sum_S_cm - (S_tp_1 + S_fn_1 + S_fp_1)
print('1\t', S_tp_1, '\t', S_fn_1, '\t', S_fp_1, '\t', S_tn_1)
S_tp_2 = S_cm[2, 2]
S_fn_2 = sum(S_cm[2]) - S_tp_2
S_fp_2 = sum(S_cm[:, 2]) - S_tp_2
S_tn_2 = sum_S_cm - (S_tp_2 + S_fn_2 + S_fp_2)
print('2\t', S_tp_2, '\t', S_fn_2, '\t', S_fp_2, '\t', S_tn_2)
S_tp_3 = S_cm[3, 3]
S_fn_3 = sum(S_cm[3]) - S_tp_3
S_fp_3 = sum(S_cm[:, 3]) - S_tp_3
S_tn_3 = sum_S_cm - (S_tp_3 + S_fn_3 + S_fp_3)
print('3\t', S_tp_3, '\t', S_fn_3, '\t', S_fp_3, '\t', S_tn_3)
S_tp_4 = S_cm[4, 4]
S_fn_4 = sum(S_cm[4]) - S_tp_4
S_fp_4 = sum(S_cm[:, 4]) - S_tp_4
S_tn_4 = sum_S_cm - (S_tp_4 + S_fn_4 + S_fp_4)
print('4\t', S_tp_4, '\t', S_fn_4, '\t', S_fp_4, '\t', S_tn_4)

S_tp = S_tp_0 + S_tp_1 + S_tp_2 + S_tp_3 + S_tp_4
S_fn = S_fn_0 + S_fn_1 + S_fn_2 + S_fn_3 + S_fn_4
S_fp = S_fp_0 + S_fp_1 + S_fp_2 + S_fp_3 + S_fp_4
S_tn = S_tn_0 + S_tn_1 + S_tn_2 + S_tn_3 + S_tn_4
print('\nTotal\t', S_tp, '\t', S_fn, '\t', S_fp, '\t', S_tn)


# In[159]:


#Calculating the performance on different parameters
S_accuracy = accuracy_score(S_y_test, S_y_pred)
print("Accuracy is ", round(S_accuracy, 2))
S_precision = precision_score(S_y_test, S_y_pred, average='weighted')
print("Precision is ", round(S_precision, 2))
S_recall = recall_score(S_y_test, S_y_pred, average='weighted')
print("Recall is ", round(S_recall, 2))
S_specificity = S_tn / (S_tn + S_fp)
print("Specificity is ", round(S_specificity, 2))
S_f1 = f1_score(S_y_test, S_y_pred, average='weighted')
print("F1 is ", round(S_f1, 2))


# In[160]:


print("\t\tDepression\tAnxiety \tStress")
print("Accuracy\t", round(D_accuracy, 2), "\t\t", round(A_accuracy, 2), "\t\t", round(S_accuracy, 2))
print("Precision\t", round(D_precision, 2), "\t\t", round(A_precision, 2), "\t\t", round(S_precision, 2))
print("Recall\t\t", round(D_recall, 2), "\t\t", round(A_recall, 2), "\t\t", round(S_recall, 2))
print("Specificity\t", round(D_specificity, 2), "\t\t", round(A_specificity, 2), "\t\t", round(S_specificity, 2))
print("F1\t\t", round(D_f1, 2), "\t\t", round(A_f1, 2), "\t\t", round(S_f1, 2))



# In[237]:


# knn method
from sklearn.neighbors import KNeighborsClassifier


# In[236]:


clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(D_X_train,D_y_train)


# In[149]:


D_y_pred=clf.predict(D_X_test)


# In[155]:


clf.score(D_X_test,D_y_test)


# In[160]:


print (classification_report(D_y_test,D_y_pred))


# In[162]:


D_cm=confusion_matrix(D_y_test,D_y_pred)
D_cm


# In[170]:


D_accuracy=accuracy_score(D_y_test,D_y_pred)
print("accuracy is", round(D_accuracy,2))
D_precision=precision_score(D_y_test,D_y_pred, average='weighted')
print("precision is", round(D_precision,2))
D_recall=recall_score(D_y_test,D_y_pred, average='weighted')
print("recall is", round(D_recall,2))
D_f1=f1_score(D_y_test,D_y_pred, average='weighted')
print("f1 score is", round(D_f1,2))


# In[196]:


clf_A= KNeighborsClassifier(n_neighbors=3)
clf_A.fit(A_X_train,A_y_train)


# In[197]:


A_y_pred=clf_A.predict(A_X_test)


# In[198]:


clf_A.score(A_X_test,A_y_test)


# In[190]:


print (classification_report(A_y_test,A_y_pred))


# In[191]:


A_cm=confusion_matrix(A_y_test,A_y_pred)
A_cm


# In[192]:


A_accuracy=accuracy_score(A_y_test,A_y_pred)
print("accuracy is", round(A_accuracy,2))
A_precision=precision_score(A_y_test,A_y_pred, average='weighted')
print("precision is", round(D_precision,2))
A_recall=recall_score(A_y_test,A_y_pred, average='weighted')
print("recall is", round(A_recall,2))
A_f1=f1_score(A_y_test,A_y_pred, average='weighted')
print("f1 score is", round(A_f1,2))


# In[230]:


clf_S= KNeighborsClassifier(n_neighbors=3)
clf_S.fit(S_X_train,S_y_train)


# In[231]:


S_y_pred=clf_S.predict(S_X_test)


# In[232]:


clf_S.score(S_X_test,S_y_test)


# In[233]:


print (classification_report(S_y_test,S_y_pred))


# In[234]:


S_accuracy=accuracy_score(S_y_test,S_y_pred)
print("accuracy is", round(S_accuracy,2))
S_precision=precision_score(S_y_test,S_y_pred, average='weighted')
print("precision is", round(S_precision,2))
S_recall=recall_score(S_y_test,S_y_pred, average='weighted')
print("recall is", round(S_recall,2))
S_f1=f1_score(S_y_test,S_y_pred, average='weighted')
print("f1 score is", round(S_f1,2))


# In[238]:


# naive byes


# In[248]:


from sklearn.naive_bayes import GaussianNB
clf_N_D=GaussianNB()
clf_N_D.fit(D_X_train,D_y_train)


# In[255]:


clf_N_D.score(D_X_test,D_y_test)


# In[256]:


D_y_pred=clf_N_D.predict(D_X_test)


# In[257]:


print (classification_report(D_y_test,D_y_pred))


# In[258]:


#using multinomial nave bayes


# In[272]:


# for anxiety
clf_N_A=GaussianNB()


# In[273]:


clf_N_A.fit(A_X_train,A_y_train)


# In[274]:


clf_N_A.score(A_X_test,A_y_test)


# In[279]:


A_y_pred=clf_N_A.predict(A_X_test)


# In[280]:


print (classification_report(A_y_test,A_y_pred))


# In[276]:


# for stress


# In[277]:


clf_N_S=GaussianNB()


# In[278]:


clf_N_S.fit(S_X_train,S_y_train)


# In[281]:


S_y_pred=clf_N_S.predict(S_X_test)


# In[282]:


clf_N_S.score(S_X_test,S_y_test)


# In[283]:


print (classification_report(S_y_test,S_y_pred))


# In[ ]:

pickle.dump(clf, open('D_model.pkl', 'wb'))
pickle.dump(A_clf, open('A_model.pkl', 'wb'))
pickle.dump(S_svm, open('S_model.pkl', 'wb'))



