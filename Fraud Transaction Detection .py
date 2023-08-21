#!/usr/bin/env python
# coding: utf-8

# # import important Library

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# # Read Datasets 

# In[2]:


data = pd.read_csv('creditcard.csv')
data


# # basic imformation about dataset

# In[3]:


data.info()


# In[4]:


data.dtypes


# In[5]:


data.shape


# In[6]:


data.index


# In[7]:


data.columns


# # Graphical Representation of Dataset

# In[8]:


fraud_data = data[data['Class'] == 1]

non_fraud_data = data[data['Class'] == 0]  

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.scatter(fraud_data['V1'], fraud_data['V2'], c='red', label='Fraud', alpha=0.5)
plt.scatter(non_fraud_data['V1'], non_fraud_data['V2'], c='blue', label='Non-Fraud', alpha=0.5)
plt.title('Scatter Plot of Fraud vs. Non-Fraud Transactions')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend()


# In[9]:


plt.subplot(1, 2, 2)
plt.hist(fraud_data['V1'], bins=50, alpha=0.5, color='red', label='Fraud')
plt.hist(non_fraud_data['V1'], bins=50, alpha=0.5, color='blue', label='Non-Fraud')
plt.xlabel('V1')
plt.ylabel('Frequency')
plt.title('Histogram of V1 for Fraud vs. Non-Fraud Transactions')
plt.legend()

plt.tight_layout()
plt.show()


# In[10]:


data = pd.read_csv('creditcard.csv')


fraud_data = data[data['Class'] == 1]


plt.figure(figsize=(10, 6))

plt.hist(fraud_data['V1'], bins=50, alpha=0.5, color='red', label='Fraud')
plt.xlabel('V1')
plt.ylabel('Frequency')
plt.title('Histogram of V1 for Fraud Transactions')
plt.legend()

plt.show()



# In[11]:


# Heatmap to visualize correlation between features
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.show()


# In[12]:


# Distribution of 'Class' (fraud or non-fraud)
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Class')
plt.title('Distribution of Class (0: Non-Fraud, 1: Fraud)')
plt.show()


# In[13]:


# Pairplot to visualize relationships between features
sns.pairplot(data, hue='Class', vars=['Time', 'Amount', 'V1', 'V2', 'V3'], plot_kws={'alpha': 0.5})
plt.show()


# In[14]:


plt.figure(figsize=(8, 6))
plt.scatter(data['V1'], data['V2'], c=data['Class'], cmap='coolwarm', alpha=0.5)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Scatter Plot of V1 vs. V2 (Color-coded by Class)')
plt.show()


# # Step 2: Data Preprocessing
# # Handling class imbalance by oversampling the minority class

# In[15]:


fraud_data = data[data['Class'] == 1]
non_fraud_data = data[data['Class'] == 0].sample(n=len(fraud_data), random_state=42)
balanced_data = pd.concat([fraud_data, non_fraud_data])

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']


# # Step 3: Feature Scaling

# In[16]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # Step 4: Split the Data Using Stratified K-Fold

# In[17]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# # Step 5: Model Selection and Hyperparameter Tuning

# In[18]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_rf_model


# # Step 6: Model Evaluation

# In[19]:


y_pred = best_rf_model.predict(X_test)
y_proba = best_rf_model.predict_proba(X_test)[:, 1]

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)


# In[20]:


classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)


# In[21]:


roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)






