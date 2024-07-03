#!/usr/bin/env python
# coding: utf-8

# # Import the Libraries

# In[1]:


#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# # Read the dataset

# In[2]:


# Read the dataset
df=pd.read_csv('dataset4.csv')
df


# In[3]:


# Plot the dataset
df_1=df.drop('category',axis=1)
df_2=df_1.drop('y', axis=1)
df_2.plot()


# In[4]:


#Describe the dataset statistically
df_2.describe()


# In[5]:


# Correlation between sensors 
df_2.corr()


# # Start clustering

# In[6]:


# Separate features and target column
X = df_1.drop('y', axis=1)  # Replace 'target_column_name' with your actual target column name
y = df_1['y']

#  Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[7]:


#  Apply K-means clustering

kmeans = KMeans(n_clusters=5)  # Define the number of clusters
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Plot the data points colored by their category
plt.figure(figsize=(10, 4))
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=df['category'], palette='Set1')
plt.title('Data Points Colored by Category by k means clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[8]:


# Apply DBSCAN clustering

db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X_scaled)
db_labels = db.labels_

# Plot the data points colored by their category
plt.figure(figsize=(10, 4))
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=df['category'], palette='Set1')
plt.title('Data Points Colored by DBSCAN Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[9]:


# Apply hierarchical clustering

hc = AgglomerativeClustering(n_clusters=5)
hc.fit(X_scaled)
hc_labels = hc.labels_

# Plot the data points colored by their category
plt.figure(figsize=(10, 4))
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=df['category'], palette='Set1')
plt.title('Data Points Colored by Hierarchical Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# # Start Dimensionality Reduction

# In[10]:


# apply pca

pca = PCA(n_components=8)

# Fit the PCA model to the data
pca.fit(df_2)

# Get the explained variance ratio for each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Get the principal components
print("Principal Components:", pca.components_)


# In[11]:


# PCA Correlate MAtrix
# Get the number of PCA components
num_components = pca.n_components_

# Get the PCA correlation matrix
pca_correlation_matrix = pd.DataFrame(pca.components_, columns=df_2.columns)

# Print the PCA correlation matrix
print("PCA Correlation Matrix:")
pca_correlation_matrix


# In[12]:


# define cumulative sum of pca
np.cumsum(pca.explained_variance_ratio_)


# In[13]:


#  apply pca on this clustered data with pca varience

import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Plot the data points colored by their category after PCA
plt.figure(figsize=(10, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['category'], palette='Set1')
plt.title('Data Points Colored by Category after PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

explained_variance = pca.explained_variance_ratio_
# Add labels to the x and y axes
plt.text(0.95, 0.05, 'Explained Variance Ratio on PCA 1: {:.2f}%'.format(explained_variance[0]*100), horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, 'Explained Variance Ratio on PCA 2: {:.2f}%'.format(explained_variance[1]*100), horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)

plt.show()


# In[14]:


# Apply LDA 

import matplotlib.pyplot as plt

# Step 1: Separate features and target column
X = df_1.drop('y', axis=1)
y = df_1['y']

# Step 2: Apply Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=4)
X_lda = lda.fit_transform(X, y)

# Step 3: Calculate the explained variance ratio
explained_variance = lda.explained_variance_ratio_
#print('% Explained Variance Ratio:', explained_variance*100)

# Step 4: Plot the first two linear discriminants
plt.figure(figsize=(10, 4))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=df['category'], palette='Set1')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('Data Points after LDA')
plt.legend()
# plt.show()

# Add labels to the x and y axes
plt.text(0.95, 0.05, 'Explained Variance Ratio on LD 1: {:.2f}%'.format(explained_variance[0]*100), horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
plt.text(0.95, 0.10, 'Explained Variance Ratio on LD 2: {:.2f}%'.format(explained_variance[1]*100), horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)

plt.show()


# In[15]:


np.cumsum(lda.explained_variance_ratio_)


# # Starting training the ML Algorithm without PCA and LDA

# In[124]:


# defining feature matrix(X) and response vector(y)
X = df_1.drop('y', axis=1)
y = df_1['y']


# In[130]:


#  apply Linear regression

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.35, random_state=42)
# Create a pipeline for Linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))


# In[133]:


#  apply Logistic regression
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.62, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))

report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}\n")


# In[136]:


#  apply Decision tree
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.53, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))

report = classification_report(y_test, y_pred)

print(f"% Accuracy: {accuracy*100}")
print(f"Classification Report:\n{report}\n")


# In[141]:


#  apply random forest
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.75, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))

report = classification_report(y_test, y_pred)

print(f"% Accuracy: {accuracy*100}")
print(f"Classification Report:\n{report}\n")


# In[142]:


#  apply KNN
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.657, random_state=42)

# Create a pipeline for KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))

report = classification_report(y_test, y_pred)

print(f"% Accuracy: {accuracy*100}")
print(f"Classification Report:\n{report}\n")


# In[147]:


#  apply PNN
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.83, random_state=42)

# Create a pipeline for PNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(hidden_layer_sizes=(3,), activation='identity', solver='lbfgs', random_state=42))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

# Print the accuracy
print('Accuracy:', accuracy)
print("training samples",len(X_train))
print("testing samples",len(X_test))

report = classification_report(y_test, y_pred)

print(f"% Accuracy: {accuracy*100}")
print(f"Classification Report:\n{report}\n")


# In[ ]:




