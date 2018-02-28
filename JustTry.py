import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
#from sklearn import linear_model
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import load_boston


#boston = load_boston()

df=pd.read_csv("abalone.data",sep=",")
#df=pd.read_csv("abalone.data",sep=",",index_col=0)
le=LabelEncoder()
le.fit(df['sex'])
df['sex']=le.transform(df['sex'])

X=df.iloc[:,0:8]
Y=df.iloc[:,8]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
      #   'learning_rate': 0.01, 'loss': 'ls'}
#dt = DecisionTreeRegressor()
#dt = RandomForestRegressor(n_estimators=150, min_samples_split=2)
linReg=LinearRegression()
#model = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
#linReg = GradientBoostingRegressor(**params)

linReg.fit(x_train,y_train)

result=linReg.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,result)))
