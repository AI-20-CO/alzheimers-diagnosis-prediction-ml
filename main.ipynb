Appendices: Python Codes
# Imports
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

# Preprocessing
df = pd.read_csv("alzheimers_disease_data.csv", encoding="utf-8")
df.head()
len(df)

# We drop irrelevant columns
print(f"Columns before drop: {df.columns.values.tolist()}")
df.drop(["PatientID", "DoctorInCharge", "Ethnicity", "EducationLevel"], inplace=True, axis=1)
print(f"Columns after drop: {df.columns.values.tolist()}")
columns_habitual = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL', "DiastolicBP", 'SystolicBP', 'Diagnosis']
df_habitual = df[columns_habitual]
columns_habitual.remove('Diagnosis')
df_chronic = df.drop(columns_habitual, axis=1)
 
print(f"Len of df_habitual columns={len(df_habitual.columns)} | Len of df_chronic columns={len(df_chronic.columns)}")
df_habitual.head()
df_chronic.head()

# Data Statistics
### Habitual
fig, axs = plt.subplots(3, 5, figsize=(24, 20))
i= 0
j = 0
for col in df_habitual.columns:
	if j == 5:
    	j = 0
    	i += 1
	if col != 'Diagnosis':
        sb.boxplot(df_habitual, x=col, hue='Diagnosis', ax=axs[i, j])
    	j += 1
fig, axs = plt.subplots(3, 5, figsize=(24, 20))
i= 0
j = 0
for col in df_habitual.columns:
	if j == 5:
    	j = 0
    	i += 1
	if col != 'Diagnosis':
        sb.kdeplot(df_habitual, x=col, hue='Diagnosis', ax=axs[i, j])
    	j += 1
### Chronic
fig, axs = plt.subplots(3, 5, figsize=(24, 20))
i= 0
j = 0
for col in df_chronic.columns:
	if j == 5:
    	j = 0
    	i += 1
	if col != 'Diagnosis':
        sb.countplot(df_chronic, x=col, hue='Diagnosis', ax=axs[i, j])
    	j += 1
fig, axs = plt.subplots(3, 5, figsize=(24, 20))
i= 0
j = 0
for col in df_chronic.columns:
	if j == 5:
    	j = 0
    	i += 1
	if col != 'Diagnosis':
        sb.kdeplot(df_chronic, x=col, hue='Diagnosis', ax=axs[i, j])
    	j += 1

# Correlation Matrix (Heatmap)
### Habitual
# Get correlation
corr = df_habitual.corr()
column_names = corr.index.values.tolist()
corr_with_diagnosis_by_strength = corr.iloc[:,-1].values.tolist()
maximum_corr_by_each_value = sorted(list(zip(column_names, corr_with_diagnosis_by_strength)), key=lambda x: x[1], reverse=True)
print("Correlation of every independent variable with <Diagnosis> in descending order:")
for name, val in maximum_corr_by_each_value:
	if name != "Diagnosis":
        print(f"{name} : {val}")
 
 
# Plot
fig, ax = plt.subplots(figsize=(36, 36))
sb.heatmap(corr, cmap="YlGnBu", annot=True, ax=ax, fmt=".3f").set_title("Correlation matrix with habitual factors")

### Chronic
# Get correlation
corr = df_chronic.corr()
column_names = corr.index.values.tolist()
corr_with_diagnosis_by_strength = corr.iloc[:,-1].values.tolist()
maximum_corr_by_each_value = sorted(list(zip(column_names, corr_with_diagnosis_by_strength)), key=lambda x: x[1], reverse=True)
print("Correlation of every independent variable with <Diagnosis> in descending order:")
for name, val in maximum_corr_by_each_value:
	if name != "Diagnosis":
        print(f"{name} : {val}")
 
 
# Plot
fig, ax = plt.subplots(figsize=(36, 36))
sb.heatmap(corr, cmap="YlGnBu", annot=True, ax=ax, fmt=".3f").set_title("Correlation matrix with chronic factors")
# Scatter plot
### Habitual
sb.pairplot(df_habitual, hue ='Diagnosis')

### Chronic
sb.pairplot(df_chronic, hue ='Diagnosis')

# Prediction with Logistic Regression
### Habitual
# Convert data to numpy arrays
X = np.array(df_habitual.iloc[:,:-1].values.tolist())
Y = np.array(df_habitual.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of logistic regression predictions (Habitual)")

### Chronic
# Convert data to numpy arrays
X = np.array(df_chronic.iloc[:,:-1].values.tolist())
Y = np.array(df_chronic.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of logistic regression predictions (Chronic)")

# Prediction with Multinomial Naive Bayes
### Habitual
# Convert data to numpy arrays
X = np.array(df_habitual.iloc[:,:-1].values.tolist())
Y = np.array(df_habitual.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of multinomialnb predictions (Habitual)")

### Chronic
# Convert data to numpy arrays
X = np.array(df_chronic.iloc[:,:-1].values.tolist())
Y = np.array(df_chronic.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of multinomialnb predictions (Chronic)")
# Prediction with Random Forest
### Habitual
# Convert data to numpy arrays
X = np.array(df_habitual.iloc[:,:-1].values.tolist())
Y = np.array(df_habitual.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of random forest predictions (Habitual)")

### Chronic
# Convert data to numpy arrays
X = np.array(df_chronic.iloc[:,:-1].values.tolist())
Y = np.array(df_chronic.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of random forest predictions (Chronic)")

# Extra: Making use of all columns i.e. Habitual + Chronic with Random Forest
# Convert data to numpy arrays
X = np.array(df.iloc[:,:-1].values.tolist())
Y = np.array(df.iloc[:, -1].values.tolist())
print(f"Shape of X={X.shape} | Shape of Y={Y.shape}")
 
# Split into train-test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
print(f"Shape of x_train,y_train={x_train.shape, y_train.shape} | Shape of x_test,y_test={x_test.shape, y_test.shape}")
 
# Build model, predict, and get scores
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
sb.heatmap(confusion_matrix(y_test, y_pred), cmap="YlGnBu", annot=True, fmt=".3g").set_title("Confusion matrix of random forest predictions (Habitual+Chronic)")
# Results Summary
N = 3
ind = np.arange(N) 
width = 0.25
 
xvals = [0.79, 0.70, 0.83]
bar1 = plt.bar(ind, xvals, width, color = 'r')
 
yvals = [0.75, 0.74, 0.72]
bar2 = plt.bar(ind+width, yvals, width, color='g')
 
plt.xlabel("Model")
plt.ylabel('Accuracy')
plt.title("Model accuracy")
 
plt.xticks(ind+width,['Logistic Regression', 'MultinomialNB', 'Random Forest'])
plt.legend( (bar1, bar2), ('Habitual', 'Chronic'))
plt.show()
 
