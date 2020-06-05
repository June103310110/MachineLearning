#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('ML_credit_card_train.csv')
print(len(df))
df2 = pd.read_csv('ML_credit_card_test.csv')
print(len(df2))

df = pd.concat([df, df2])
df


# In[3]:


df.isnull().any()


# In[4]:


a = str(df['country'].mode())
print(a)
df.fillna(a, inplace = True)
df.isnull().any()


# In[5]:


df


# In[6]:


# print(df.columns)
# # drop_lis = ['user_id', 'device_id', 'signup_time', 'purchase_time']
# drop_lis = ['signup_time']
# data = df.drop(columns = drop_lis)
# print(data.columns)


# In[7]:


strange_lis = ['source', 'browser', 'country', 'user_id', 'device_id']
data2 = df.copy()

from sklearn import preprocessing


def label_enc(data):
    le = preprocessing.LabelEncoder()
    le.fit(data)
    arr = le.transform(data)
    data = pd.DataFrame(arr)
    return data

for i in strange_lis:
    data2[i] = label_enc(data2[i])


# In[8]:


data2


# In[9]:


data2.reset_index(drop=True, inplace =True)


# In[10]:


data2['signup_time'][0][8:10]


# In[11]:


col = ['signup_time']
def new_col_of_time(data2, col):
    lis_hour = []
    lis_month =[]
    lis_day =[]
    for i in range(len(data2)):
        hour = int(data2[col][i][-8:-6])
        month = int(data2[col][i][5:7])
        day = int(data2[col][i][8:10])
        lis_hour.append(hour)
        lis_month.append(month)
        lis_day.append(day)

    b = 'new_'+col+'hour'
    data2[b] = pd.DataFrame(lis_hour)
    
    b = 'new_'+col+'month'
    data2[b] = pd.DataFrame(lis_month)
    
    b = 'new_'+col+'day'
    data2[b] = pd.DataFrame(lis_day) 

new_col_of_time(data2, col[0]) 
data2.pop(col[0])
data2


# In[12]:


lis = []
for i in range(len(data2)):
    a = int(data2['purchase_time'][i][-8:-6])
    lis.append(a)
#     print(type())
#     data2.at[i,'purchase_time'] = int(a)
data2['new_purchase_time'] = pd.DataFrame(lis)
data2.pop('purchase_time')
data2


# In[13]:


X_test = data2[-27200:]
y_test = X_test.pop('class')
print(len(X_test))
X_train = data2[:-27200]
print(len(X_train))
y_train = X_train.pop('class')


# In[14]:


print(sum(y_train), len(y_train))
a = len(y_train)-sum(y_train)
b = a/sum(y_train)
b


# In[15]:



data2.select_dtypes(include='object')


# In[16]:


from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import time;  # 引入time模块


stratified_folder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)
for train_index, test_index in stratified_folder.split(X_train, y_train):
#     print("Stratified Train Index:", train_index)
#     print("Stratified Test Index:", test_index)
#     a = sum(y_train[train_index])/len(y_train[train_index])
#     b = sum(y_train[test_index])/len(y_train[test_index])
#     print("Stratified y_train:", a, sum(y_train[train_index]))
#     print("Stratified y_test:", b, sum(y_train[test_index]), '\n')
    
    X = X_train.loc[train_index, :] # 有很多個columns 要指定全部
    y = y_train.loc[train_index] # 只有一個columns
    
    Xt = X_train.loc[test_index, :]
    yt = y_train.loc[test_index]
    
    lis = [2, 4]
    for i in lis:
        model = XGBClassifier(max_depth= i, learning_rate=.005, n_estimators = 3000,
                              tree_method = 'gpu_hist', predictor = 'gpu_predictor',
                              random_state =12, scale_pos_weight = 9.738255033557047,
        #                     reg_lambda = 0.2
                             )

        # 模型 訓練
        # clf.fit(X_train,y_train,eval_metric='auc')
        model.fit(X=X,y=y, eval_set=[(X, y), (Xt, yt)],
                  eval_metric = ['error', 'auc',],
                  verbose = True
                 )


        # 預測值
        y_pred=model.predict(Xt)
        # 真實值 賦值
        y_true= yt

        # 計算精度
        print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
        print(metrics.precision_recall_fscore_support(y_true, y_pred))
        print(metrics.f1_score(y_true, y_pred))
        print(metrics.classification_report(y_true, y_pred))

        line = str(metrics.precision_recall_fscore_support(y_true, y_pred))+'\n\n'
        line1 = str(model.get_params)



        localtime = time.asctime( time.localtime(time.time()) )
        ticks = str(localtime)+'\n'

        fp = open("save.txt", "a")

        # 寫入 This is a testing! 到檔案
        # lines = ["One\n", "Two\n", "Three\n", "Four\n", "Five\n"]
        segline = '\n------------------------------------\n'
        fp.writelines(segline)
        fp.writelines(ticks)
        fp.writelines(line)
        fp.writelines(line1)

        # 關閉檔案
        fp.close()


# In[ ]:




