#  GRIP  Task 1- Prediction using Supervised ML 

Predict the percentage of an student based on the no. of study hours. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
%matplotlib inline

#loading the dataset
url = "http://bit.ly/w-data"
data_set = pd.read_csv(url)
print("Data imported successfully")
data_set.head()

#plotting the distribution of dataset
data_set.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.show()

data_set.describe()

preparing the data


#checking there is any missing values
data_set['Hours'].isnull().sum()
data_set['Scores'].isnull().sum()

data_set.shape

from scipy import stats
z1=np.abs(stats.zscore(data_set.Scores))
print(z1)

threshold=2
print(np.where(z1>2))

z2=np.abs(stats.zscore(data_set.Hours))
print(z2)

threshold=2
print(np.where(z2>2))

outlier analysis using boxplot


data1=data_set['Scores']
fig=plt.figure(figsize=(10,5))
plt.boxplot(data1)
plt.show()

data2=data_set['Hours']
fig=plt.figure(figsize=(10,5))
plt.boxplot(data2)
plt.show()

from scipy.stats import skew
skew(data_set)


#checking the correlation
data_set.corr()


X = data_set.iloc[:, :-1].values  
y = data_set.iloc[:, 1].values

#split the data into train and test set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)

Training the algorithm

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

Making predictions

print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# You can also test with your own data
hours = [[9.25]]
print("number of hours:9.25")
print("Score:")
own_pred = regressor.predict([[9.25]])
own_pred

Evaluating the model

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

