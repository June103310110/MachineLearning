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
df0 = X_train[X_train['class'] == 0]
df1 = X_train[X_train['class'] == 1]
df = df0.sample(10000)
X_train = pd.concat([df, df1], axis = 0)

y_train = X_train.pop('class')

# X_train = X_train[:10000]

# In[14]:


print(sum(y_train), len(y_train))
a = len(y_train)-sum(y_train)
b = a/sum(y_train)
b


# In[15]:



data2.select_dtypes(include='object')


# In[67]:


from sklearn import metrics
from xgboost.sklearn import XGBClassifier

lis1 = [12]
for _ in lis1:
    model = XGBClassifier(
    )

    # 模型 訓練
    # clf.fit(X_train,y_train,eval_metric='auc')
    model.fit(X=X_train,y=y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose = True)

    # 預測值
    y_pred=model.predict(X_test)
    # 真實值 賦值
    y_true= y_test

    # 計算精度
    print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
    print(metrics.precision_recall_fscore_support(y_true, y_pred))
    print(metrics.f1_score(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred))

    line = str(metrics.precision_recall_fscore_support(y_true, y_pred))+'\n\n'
    line1 = str(model.get_params)
    line2 = '\n'+str(sum(y_train))+','+str(len(y_train))
    from sklearn import metrics

    # 取得預測值
    y_pred=model.predict(X_test)
    # 賦值給真實值 
    y_true= y_test

    # 計算預測的 Precision, Recall, f1-score 以及在測試資料集中各類別的總量
    print(metrics.precision_recall_fscore_support(y_true, y_pred))


    lis = ['precision', 'recall', 'f1-score', 'classes-numbers']
    arr = metrics.precision_recall_fscore_support(y_true, y_pred)
    df = pd.DataFrame(arr)
    df.rename(columns={df.columns[0]: "class0", df.columns[1]: "class1"}, inplace = True)
    df['metric'] = pd.DataFrame(lis)
    df


    import time;  # 引入time模块
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
    fp.writelines(line2)

    # 關閉檔案
    fp.close()
    
# 學太快了
print(model.get_params)




