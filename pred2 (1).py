import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

train_dataset = pd.read_excel(r'D:\ver\Data_Train (1).xlsx')
test_data = pd.read_excel(r'D:\ver\Data_Test (1).xlsx')
test_dataset=test_data[:]
pd.set_option("display.max_column", 50)
pd.set_option('display.width', 2000)

train_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r'[a-z]*[/][a-z]*', value=r'')
test_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r'[a-z]*[/][a-z]*', value=r'')
train_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r'[a-z]*', value=r'')
test_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r'[a-z]*', value=r'')
train_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r"[' ']*", value=r'')
test_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r"[' ']*", value=r'')
train_dataset['Engine'].replace(regex=True, inplace=True, to_replace=r"[' ']['CC']*", value=r'')
test_dataset['Engine'].replace(regex=True, inplace=True, to_replace=r"[' ']['CC']*", value=r'')
train_dataset['Power'].replace(regex=True, inplace=True, to_replace=r"[' ']['bhp']*", value=r'')
test_dataset['Power'].replace(regex=True, inplace=True, to_replace=r"[' ']['bhp']*", value=r'')


train_dataset['Mileage'].fillna(train_dataset['Mileage'].median(), inplace=True)
test_dataset['Mileage'].fillna(train_dataset['Mileage'].median(), inplace=True)
train_dataset['Engine'].fillna(train_dataset['Engine'].median(), inplace=True)
test_dataset['Engine'].fillna(train_dataset['Engine'].median(), inplace=True)
# filling nan with 74.0 in POWER feature because it is the most frequent value in that feature
train_dataset['Power'].fillna(74.0, inplace=True)
test_dataset['Power'].fillna(74.0, inplace=True)

# li=train_dataset['Power'].isnull()
train_dataset['Power'].replace(regex=True, inplace=True, to_replace=r"['null']", value=74.0)
test_dataset['Power'].replace(regex=True, inplace=True, to_replace=r"['null']", value=74.0)
train_dataset['Engine'].replace(regex=True, inplace=True, to_replace=r"['null']", value=1346.0)
test_dataset['Engine'].replace(regex=True, inplace=True, to_replace=r"['null']", value=1346.0)
train_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r"['null']", value=21.5)
test_dataset['Mileage'].replace(regex=True, inplace=True, to_replace=r"['null']", value=21.5)

# 5.0 is the most frequent value in seats feature
train_dataset['Seats'].fillna(5.0, inplace=True)
test_dataset['Seats'].fillna(5.0, inplace=True)
train_dataset = train_dataset.drop("New_Price", axis=1)
test_dataset = test_dataset.drop("New_Price", axis=1)
train_dataset = train_dataset.drop("Name", axis=1)
test_dataset = test_dataset.drop("Name", axis=1)


X = train_dataset
X1 = test_dataset

le = LabelEncoder()

X['Location'] = le.fit_transform(X['Location'])
X1['Location'] = le.fit_transform(X1['Location'])
X['Year'] = le.fit_transform(X['Year'])
X1['Year'] = le.fit_transform(X1['Year'])
X['Fuel_Type'] = le.fit_transform(X['Fuel_Type'])
X1['Fuel_Type'] = le.fit_transform(X1['Fuel_Type'])
X['Transmission'] = le.fit_transform(X['Transmission'])
X1['Transmission'] = le.fit_transform(X1['Transmission'])
X['Owner_Type'] = le.fit_transform(X['Owner_Type'])
X1['Owner_Type'] = le.fit_transform(X1['Owner_Type'])
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X1 = onehotencoder.fit_transform(X1).toarray()
#%%
onehotencoder1 = OneHotEncoder(categorical_features=[13])
X = onehotencoder1.fit_transform(X).toarray()
X1 = onehotencoder1.transform(X1).toarray()
#%%
X = X[:, 0:24]
y = train_dataset.iloc[:, -1].values
# %%


X_train, X_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size=0.2, random_state=0)

linreg = LinearRegression()

polynomial_features = PolynomialFeatures(degree=3)
polynomial_features.fit(X_train)
x_poly = polynomial_features.transform(X_train)
x_polyt = polynomial_features.transform(X_test)

linreg.fit(x_poly, y_train)
y_pred = linreg.predict(x_polyt)
print(f'R2 with POlyRegression of 3 Degree: {r2_score(y_test, y_pred)}')
print("Final RMSE value is =", np.sqrt(np.mean((y_test - y_pred) ** 2)))

x_poly = polynomial_features.transform(X1)
y_pred = linreg.predict(x_poly)
#%%
test_data["Price"] = y_pred
#%%
test_data.to_csv(r"D:\ver\Predicted_test.csv")
#%%
test_data.to_excel(r"D:\ver\Predicted_test.xlsx")