#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#Pre_processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#Errors
from sklearn.metrics import mean_absolute_error , mean_squared_error , mean_absolute_percentage_error , r2_score
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Loading the dataset
df = sns.load_dataset('diamonds')

#Separate X and y
X = df.drop('price' , axis=1)
y = df['price']

#Categories
Numeric_features = ['carat','depth','table','x','y','z']
categorical_features = ['cut','color','clarity']


#preprocessing
pre_processing = ColumnTransformer([
    ('num' , StandardScaler() , Numeric_features),
    ('cat' , OneHotEncoder(), categorical_features)
])

#Train , Test and Split data 
X_train , X_test , y_train , y_test = train_test_split(X , y , train_size=0.8 , random_state=42)


#Build the pipeline
pipeline = Pipeline([
    ('processing' , pre_processing),
    ('model' , XGBRegressor())
])

#Training the model
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
#Evaluation
print('MAE:', mean_absolute_error(y_test , y_pred))
print('MSE:', mean_squared_error(y_test , y_pred))
print('RMSE', np.sqrt(mean_squared_error(y_test , y_pred)))
print('MAPE', mean_absolute_percentage_error(y_test , y_pred))
print('r2_score', r2_score(y_test, y_pred))

'''
Using Linear Regression:
    MAE: 737.1513665933284
    MSE: 1288705.477851676
    RMSE 1135.2116445190632
    MAPE 0.3952933516494359
    r2_score 0.9189331350419386

using Decision Tree Regressor:
    MAE: 358.2411475713756
    MSE: 542613.8330784205
    RMSE 736.6232640084214
    MAPE 0.08605882365467431
    r2_score 0.9658665202511018    

using Random Forest Regressor:
    MAE: 271.46964567553016
    MSE: 304820.96340730065
    RMSE 552.1059349502599
    MAPE 0.06511207295198045
    r2_score 0.9808250369835319    

using XGB-regressor:
    MAE: 285.61346435546875
    MSE: 318286.3125
    RMSE 564.1686915276316
    MAPE 0.07398492097854614
    r2_score 0.9799779653549194       
'''
# ======================================================
# ✅ Conclusion:
# Based on evaluation metrics, Random Forest performs best
# ======================================================