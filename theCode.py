#with error: 1.9576288203

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
#from sklearn import linear_model
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import load_boston


#boston = load_boston()

df=pd.read_csv("abalone.data",sep=",")
#df=pd.read_csv("abalone.data",sep=",",index_col=0)
le=LabelEncoder()
le.fit(df['sex'])
df['sex']=le.transform(df['sex'])

#change the features
#for i in df.columns:
    
    #df.fillna(-999,inplace=True)
    #col=int(math.ceil(0.01*len(df)))
    #df[i]=df[i].shift(-col)
forcast_col='length'
df.fillna(-99999,inplace=True)
forcast_out=int(math.floor(0.001*len(df)))
df['lenght2']=df[forcast_col].shift(-forcast_out)

forcast_col='diameter'
df.fillna(-99999,inplace=True)
forcast_out=int(math.floor(0*len(df)))
df['diameter2']=df[forcast_col].shift(-forcast_out)

forcast_col='height'
df.fillna(-99999,inplace=True)
forcast_out=int(math.ceil(0.01*len(df)))
df['height2']=df[forcast_col].shift(-forcast_out)

#no change
forcast_col='whole-weight'
df.fillna(-99999,inplace=True)
forcast_out=int(math.ceil(0*len(df)))
df['whole-weight2']=df[forcast_col].shift(-forcast_out)

#no change
forcast_col='shucked-weight'
df.fillna(-99999,inplace=True)
forcast_out=int(math.ceil(0*len(df)))
df['shucked-weight2']=df[forcast_col].shift(-forcast_out)

#no change
forcast_col='viscera-weight'
df.fillna(-99999,inplace=True)
forcast_out=int(math.ceil(0.000001*len(df)))
df['viscera-weight2']=df[forcast_col].shift(-forcast_out)

#no change
forcast_col='shell-weight'
df.fillna(-99999,inplace=True)
forcast_out=int(math.ceil(0*len(df)))
df['shell-weight2']=df[forcast_col].shift(-forcast_out)

X=df.iloc[:,[0,9,10,11,12,13,14,15]]
Y=df.iloc[:,8]


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=200)
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'ls'}
#dt = DecisionTreeRegressor()
dt = RandomForestRegressor(n_estimators=350, min_samples_split=30)
#linReg=LinearRegression()
#model = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
#linReg = GradientBoostingRegressor(**params)

dt.fit(x_train,y_train)

result=dt.predict(x_test)
error=np.sqrt(mean_squared_error(y_test,result))
print(error)
