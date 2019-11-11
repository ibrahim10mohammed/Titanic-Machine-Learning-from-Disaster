import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression

training_set = pd.read_csv('train.csv')
survive = training_set.Survived
#training_set = training_set.drop(['Survived'],axis=1)
test_set = pd.read_csv('test.csv')
training_set = pd.concat([training_set, test_set],ignore_index=True,sort=False)

training_set['Sex'] = training_set['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = training_set[(training_set['Sex'] == i) &(training_set
                                ['Pclass']== j+1)]['Age'].dropna()         
        age_guess = guess_df.median()
        
        guess_ages[i,j] = float (age_guess)
for i in range(0, 2):
    for j in range(0, 3):
        training_set.loc[ (training_set.Age.isnull()) & (training_set.Sex == i) & (training_set.Pclass == j+1),'Age'] = guess_ages[i,j]
training_set['Age'] = training_set['Age'].astype(float)

training_set['Age']=training_set['Age']/training_set['Age'].sum()

freq_Embarked=training_set.Embarked.dropna().mode()

training_set['Embarked'] = training_set['Embarked'].fillna(freq_Embarked[0])
training_set['Embarked'] = training_set['Embarked'].map( {'S': 0, 'C': 0.5, 'Q': 1}).astype(float)

fare_mean = training_set['Fare'].dropna()
fare_mean = fare_mean.mean()

training_set['Fare'] = training_set['Fare'].fillna(fare_mean)

training_set['Fare'] = training_set['Fare']/training_set['Fare'].sum()

training_set['Family'] = training_set['Parch']+training_set['SibSp']+1
training_set['Family'] = training_set['Family']/training_set['Family'].sum()

'''
        third version  gender and Embarked 
'''
X_train = training_set[['Sex','Embarked']]# independent variables only
X_train = X_train[:891]
Y_train = training_set['Survived'][:891]
X_test = training_set[['Sex','Embarked']]
X_test = X_test[891:]

logisticreg = LogisticRegression()
logisticreg.fit(X_train, Y_train)
Y_pred = logisticreg.predict(X_test)

training_set.Survived[891:]=Y_pred
final_result = training_set[['PassengerId','Survived']]
final_result = final_result[891:]
final_result.to_csv('gender-embarked.csv')