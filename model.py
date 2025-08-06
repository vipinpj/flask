import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv(r"C:\Users\DELL\OneDrive\salary/hiring.csv")


dataset['experience'].fillna(0,inplace=True)



X = dataset.iloc[:, :3]


def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,
                 'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return word_dict[word]
X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]




from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X,y)

#saving lr to disk
pickle.dump(lr, open('lr.pkl','wb'))


lr = pickle.load(open('lr.pkl','rb'))
print(lr.predict([[2,9,6]]))