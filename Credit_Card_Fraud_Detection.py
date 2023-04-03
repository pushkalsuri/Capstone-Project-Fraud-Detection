#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


# In[2]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv(r'C:\Users\pushk\OneDrive - Durham College\AI\Sem-2\CAPSTONE TERM II\Model\creditcard.csv')


# In[3]:


# first 5 rows of the dataset
credit_card_data.head()


# In[5]:


# dataset informations
credit_card_data.info()


# In[6]:


# checking the number of missing values in each column
credit_card_data.isnull()


# In[7]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# This Dataset is highly unblanced

# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[8]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


# statistical measures of the data
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Under-Sampling

# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

# Number of Fraudulent Transactions --> 492

# In[13]:


legit_sample = legit.sample(n=492)


# In[14]:


legit_sample.Amount.describe()


# Concatenating two DataFrames

# In[15]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[16]:


new_dataset.head()


# In[17]:


new_dataset.tail()


# In[18]:


new_dataset['Class'].value_counts()


# In[19]:


new_dataset.groupby('Class').mean()


# In[20]:


new_dataset = new_dataset.drop(columns=['Time'])


# In[21]:


# Plot a correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(new_dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')


# In[22]:


# Plot a histogram of the 'Amount' column for the legit class
plt.figure(figsize=(12, 6))
sns.histplot(data=legit, x='Amount', kde=True)
plt.title('Legit Transactions - Amount Distribution')


# In[23]:


# Plot a histogram of the 'Amount' column for the fraud class
plt.figure(figsize=(12, 6))
sns.histplot(data=fraud, x='Amount', kde=True)
plt.title('Fraud Transactions - Amount Distribution')


# Splitting the data into Features & Targets

# In[24]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[25]:


print(X)


# In[26]:


print(Y)


# Split the data into Training data & Testing Data

# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[29]:


#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)


# Model Training

# In[30]:


model = RandomForestClassifier(max_depth=5, max_features = 7, n_estimators = 10)


# In[31]:


# training the random forest with Training Data
model.fit(X_train2, Y_train)


# Model Evaluation

# Accuracy Score

# In[32]:


# accuracy on training data
X_train_prediction = model.predict(X_train2)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[33]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[34]:


# accuracy on test data
X_test_prediction = model.predict(X_test2)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[35]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[36]:


# Create the model
rf = RandomForestClassifier()
# Define the hyperparameters to be tuned
params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'max_features': ['sqrt', 'log2']
}
# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train2, Y_train)

# Get the best hyperparameters
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_search.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)


# In[37]:


# Print the confusion matrix
X_test_prediction = model.predict(X_test2)
cm = confusion_matrix(Y_test, X_test_prediction)
print('Confusion Matrix : \n', cm)


# In[ ]:

# Save the best model to a pickle file
best_model = grid_search.best_estimator_
pickle.dump(best_model, open("best_model.pkl", "wb"))



