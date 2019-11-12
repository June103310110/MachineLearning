#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


import os, sys
os.chdir( '~/git/nogitfile' )
# %cd '~/git/nogitfile'
# !ls
ori_df = pd.read_csv('train.csv')

### 看有幾個column
print(f'-共有{ori_df.shape[0]}筆資料')
#print('-共有{0}筆資料'.format(df.shape[0]))
print('-分布於%d欄'%ori_df.shape[1])
import numpy as np
def timeSinCosTrans(df):

    Time = df.loc[:,'loctm'].values
    # // 是整除的意思
    s, m, h = [(Time//100**(i)%100) for i in range(3)]
    Time = h*3600+m*60+s

    dfTime = pd.DataFrame()
    dfTime.loc[:,'TIME_SIN'] = np.sin(2*np.pi*Time/(24*60*60))
    dfTime.loc[:,'TIME_COS'] = np.cos(2*np.pi*Time/(24*60*60))
    dfTime.loc[:,'time'] = Time

    #df.drop(columns = 'loctm', inplace =True)
    df = pd.concat([df, dfTime], axis = 1)

    return df


# In[3]:


def submit(model):

    test_df = pd.read_csv('test.csv')
    #rint(len(test_df))
    ID = test_df['txkey']


   # to_drop = ['bacno','txkey','cano']
   # test_df = test_df.drop(columns=to_drop)

    #print(test_df.info())
    test_df = timeSinCosTrans(test_df)

   # print(len(test_df))
    test_df.fillna(0, inplace = True)

    test = test_df

    YN = ['ecfg','flbmk', 'flg_3dsmk', 'insfg', 'ovrlt']
    for col in YN:
        test.loc[test[col] =='N', col] = 0
        test.loc[test[col] =='Y', col] = 1


    

    label = model.predict(test)

    from datetime import datetime
    time_str = datetime.now().strftime('%m-%d:%H:%M')

    submit = {'txkey':ID, 'fraud_ind':label}
    pd.DataFrame(data= submit).to_csv('submit_'+time_str+'.csv',index = False)


# ### 基礎資料處理

# In[4]:


df = ori_df.copy()

# 把Y/N 資料轉成 0/1
YN = ['ecfg','flbmk', 'flg_3dsmk', 'insfg', 'ovrlt']
for col in YN:
    df.loc[df[col] =='N', col] = 0
    df.loc[df[col] =='Y', col] = 1
df.fillna(0, inplace =True)

# 把時間做sin/cos轉換，保留原來時間
df = timeSinCosTrans(df)


# #### 抓出所有被詐騙過的帳戶進行分析

# In[5]:


fraud_df = df.loc[df['fraud_ind'] == 1]
print("總共的詐騙次數共有%d筆"%len(fraud_df))
fraud_bacno_list = list(set(fraud_df['bacno']))

new_df = df[df['bacno'].isin(fraud_bacno_list)]

print("所有被詐騙帳戶的資料有%d筆"%len(new_df))
new_df.head()


# In[6]:


from sklearn.model_selection import train_test_split

# df = new_df.copy()
y = new_df.pop('fraud_ind')
X_data, X_test, y_data, y_test = train_test_split(new_df, y, train_size = .9, stratify = y, random_state = 20)


# In[7]:


# 訓練模型評估結果
def Cross_valid(X_data, y_data, n):
    from sklearn.model_selection import StratifiedKFold 
    import xgboost as xgb
    from sklearn.metrics import f1_score
    import sklearn.model_selection


    skf = StratifiedKFold(n_splits=n, random_state=123, shuffle=True)
    skf.get_n_splits(X_data, y_data)
    print(skf)  



    for train_index, test_index in skf.split(X_data, y_data):

        print("TRAIN:", train_index, "TEST:", test_index)  
        X_train, X_valid = X_data[X_data.index.isin(train_index)], X_data[X_data.index.isin(test_index)]
        y_train, y_valid = y_data[y_data.index.isin(train_index)], y_data[y_data.index.isin(test_index)]

    #     print(X_train)
        model = xgb.XGBClassifier(max_depth= '8', learning_rate=.05, n_estimators = 1000,                              tree_method = 'gpu_hist', predictor = 'gpu_predictor')

        model.fit(X=X_train,y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose = False)

        y_true = y_valid.values
        y_pred = model.predict(X_valid)
        print(f1_score(y_true, y_pred, average=None))
 
        


# In[8]:


def train_xgb(data, max_depth, learning_rate, n_estimators):
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    # df = new_df.copy()
    y = data.pop('fraud_ind')
    X_train, X_valid, y_train, y_valid = train_test_split(data, y, train_size = .9, stratify = y, random_state = 20)
    
    model = xgb.XGBClassifier(max_depth= max_depth, learning_rate=learning_rate, n_estimators = n_estimators,                              tree_method = 'gpu_hist', predictor = 'gpu_predictor')

    model.fit(X=X_train,y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose = False)

    y_true = y_valid.values
    y_pred = model.predict(X_valid)
    print(f1_score(y_true, y_pred, average=None))
    


# ### 訓練模型評估結果

# In[9]:


# 提交
#submit()


# ### 

# ### 分析
# 1. 有詐騙的帳戶漢沒有詐騙的帳戶之間的異同 (麻煩)
# 2. 被詐騙的帳戶資料之間的變化

# In[10]:


# 先合併一下X和Y方便分析，順便把index拔掉
data = pd.concat([X_data, y_data], axis = 1)
data.reset_index(inplace = True)

cano_list = list(set(X_data['cano']))
len(set(X_data['cano']))

lis = []
for i in cano_list:
    tdf = X_data[X_data['cano'] == i]
    lis.append([i, len(tdf)])
#         print(i,len(tdf))

fraud_cano = np.array(lis)
import numpy as np
np.mean(fraud_cano[:,1])
np.max(fraud_cano[:,1])
sns.boxplot(list(fraud_cano[:,1]), orient ='v')
np.median(fraud_cano[:,1])
# In[11]:


data.columns


# ### 分析報告
# 1. 時間沒什麼用, 大部分詐欺的發生點跟平常消費的時段差不多
# 2. 地點可能有用，但也許透過帳戶而不是卡號來分析地點會更有用 (scity), 可以加入一個消費地點屬於陌生地區的標記
# 3. 詐欺往往發生在同一個日期上，可以透過單日消費次數來判斷有詐欺風險
# 4. 3dsmk好像有用，不過很少被使用的樣子
# 5. 詐欺的總價有時是連續的小筆(金額相同或相似)(在相同日期)，也會出現明顯離群的大筆支出

# In[12]:


#從卡號觀察每張卡被刷的 單日總次數/ 單日平均次數

cano_list = list(set(X_data['cano']))
test = cano_list[:0]

for i in test:
    # i是依序被從cano_list中讀出來的卡號, tmpdf則是依據卡號抓取出來的資料，也就是分析對象(每一張卡的所有資料)
    tmp_df = data[data['cano'] == i]
    lis = ['locdt', 'cano', 'conam', 'fraud_ind', 'scity', 'hcefg', 'loctm', 'flg_3dsmk']
    print('\n----------',i)
    print(tmp_df[lis])

lis = [0,1,2,3,4,5,5,5,5]
np.argmax(np.bincount(lis))

# ### 根據上面的分析做一些特徵工程

# #### 先做一個等一下要用來填充特徵工程資料的的column, 'home_scity'

# In[13]:


# data.head()

def add_column(data, column_name):
    a = list(data.columns)
    if any(column_name in s for s in a):
        print("Column \'%s\' is already existed"%column_name)
    else:
        a = data.columns[1]
        tmp_df = data[a]
        tmp_df = tmp_df.rename(columns={a: column_name})
        data[column_name] = tmp_df
        
add_column(data, 'total_times')
add_column(data, 'days_times')
add_column(data, 'home_scity')
data.head()


# In[14]:


data_column = list(data.columns)

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(arr)
np.delete(arr, [1,2], 0)
# In[15]:


cano_list = list(set(data['cano']))


# In[16]:


# 增加交易總次數
# 在每一個交易日期加入該日期的交易次數
def trade_count_enc_cano(data):
    
    cano_list = list(set(data['cano']))
    test = cano_list[:]
    data_array = np.array(data)
    data_column = list(data.columns)
    cnt = 0

    for i in test:
        # 進度條的迭代器
        cnt+=1
    #     print(len(data))

        # i是一個一個被從cano_list中讀出來的卡號
        # a0表示這個卡號總共出現過幾次，也就是總共的交易次數
        a0 = len(data[data['cano'] == i])

    #     # col是在所有的column中尋找指定column('total_times')的index
        col = list(data.columns).index('total_times')
    #     # index則是指定所有卡號為i的資料的index
        index = data[data['cano'] == i].index
    #     print(index)
    #     # 使用numpy array的做法來把值填充進去
        data_array[index ,col] = a0

        #-----------------------------------------------
        # 在每一個交易日期加入該日期的交易次數
        lis_day = set(data[data['cano'] == i]['locdt'])
    #     print(lis_day)

    # col是在所有的column中尋找指定column('days_times')的index
        col_days_times = list(data.columns).index('days_times')
    #     print(col_days_times)
        for _ in lis_day:
            a1 = data[data['cano'] == i][data[data['cano'] == i]['locdt'] == _]

        # 在某個卡號的資料中指定某個日期的index
            index_a1 = a1.index
    #         print(index_a1)
        # 找出該日期的資料長度(交易次數)
            len_a1 = len(a1)
        # 使用numpy array的方法來把值填充進去
            data_array[index_a1, col_days_times] = len_a1

    #         print(len(a1))
    #         print(_)
        ###############################################
        # 弄一個進度條
        print(cnt/len(cano_list))
#     print(data_array[index, :])
    return data_array
# data_array = trade_count_enc_cano(data)

# 增加交易總次數
# 在每一個交易日期加入該日期的交易次數
cano_list = list(set(X_data['cano']))
test = cano_list[:]
data_array = np.array(data)
data_column = list(data.columns)
cnt = 0

for i in test:
    # 進度條的迭代器
    cnt+=1
#     print(len(data))
    
    # i是一個一個被從cano_list中讀出來的卡號
    # a0表示這個卡號總共出現過幾次，也就是總共的交易次數
    a0 = len(data[data['cano'] == i])
    
#     # col是在所有的column中尋找指定column('total_times')的index
    col = list(data.columns).index('total_times')
#     # index則是指定所有卡號為i的資料的index
    index = data[data['cano'] == i].index
#     print(index)
#     # 使用numpy array的做法來把值填充進去
    data_array[index ,col] = a0
    
    #-----------------------------------------------
    # 在每一個交易日期加入該日期的交易次數
    lis_day = set(tmp_data[tmp_data['cano'] == i]['locdt'])
#     print(lis_day)
    
# col是在所有的column中尋找指定column('days_times')的index
    col_days_times = list(data.columns).index('days_times')
#     print(col_days_times)
    for _ in lis_day:
        a1 = data[data['cano'] == i][data[data['cano'] == i]['locdt'] == _]

    # 在某個卡號的資料中指定某個日期的index
        index_a1 = a1.index
#         print(index_a1)
    # 找出該日期的資料長度(交易次數)
        len_a1 = len(a1)
    # 使用numpy array的方法來把值填充進去
        data_array[index_a1, col_days_times] = len_a1
    
#         print(len(a1))
#         print(_)
    ###############################################
    # 弄一個進度條
    print(cnt/len(cano_list))
#     print(data_array[index, :])
df2_0 = pd.DataFrame(data_array, columns = data_column)
# In[17]:


# np.savez_compressed('modify_data.npz', data = data_array)


# In[18]:


data = np.load('modify_data.npz')
data = data['data']

# # y = df2_0.pop('fraud_ind')
# # train(df2_0, y)
# data['df2_0']
data = pd.DataFrame(data, columns = data_column)
# y = data.pop('fraud_ind')
# Cross_valid(data, y, 3)
data.columns
data.drop(columns = ['index'], inplace = True)
data.columns


# In[19]:


import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

data = np.load('modify_data.npz')
data = data['data']
data = pd.DataFrame(data, columns = data_column)

# df = new_df.copy()
y = data.pop('fraud_ind')
X_train, X_valid, y_train, y_valid = train_test_split(data, y, train_size = .9, stratify = y, random_state = 20)

model = xgb.XGBClassifier(max_depth= '8', learning_rate=.05, n_estimators = 1000,                          tree_method = 'gpu_hist', predictor = 'gpu_predictor')

model.fit(X=X_train,y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose = False)

y_true = y_valid.values
y_pred = model.predict(X_valid)
print(f1_score(y_true, y_pred, average=None))


# In[ ]:


test_df = pd.read_csv('test.csv')
#rint(len(test_df))
ID = test_df['txkey']

# to_drop = ['bacno','txkey','cano']
# test_df = test_df.drop(columns=to_drop)

#print(test_df.info())
test_df = timeSinCosTrans(test_df)

# print(len(test_df))
test_df.fillna(0, inplace = True)

test = test_df

YN = ['ecfg','flbmk', 'flg_3dsmk', 'insfg', 'ovrlt']
for col in YN:
    test.loc[test[col] =='N', col] = 0
    test.loc[test[col] =='Y', col] = 1


# In[ ]:


data = test
print(len(data))
print(data.head(3))
add_column(data, 'total_times')
add_column(data, 'days_times')
add_column(data, 'home_scity')

data_column = list(set(data.columns))


# In[ ]:


test_arr = trade_count_enc_cano(data)

# data = pd.DataFrame(data)
print(len(test_arr))


# In[ ]:


test_data = pd.DataFrame(test_arr, columns = data_column)
print(data)
# data = data.drop(columns = ['fraud_ind'])


# In[ ]:


label = model.predict(data)

from datetime import datetime
time_str = datetime.now().strftime('%m-%d:%H:%M')

submit = {'txkey':ID, 'fraud_ind':label}
pd.DataFrame(data= submit).to_csv('submit_'+time_str+'.csv',index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#從卡號觀察每張卡被刷的月總次數/ 月平均單日次數/ 有刷卡日子的平均次數/ 單日總次數/ 月平均頻率/ 單日最高/ 單日最高頻率
lis = []
test = cano_list[:10]
tmp_data = data.copy()
for i in test:
    # i是一個一個被從cano_list中讀出來的卡號
    tmp_df = tmp_data[tmp_data['cano'] == i]
#     print(tmp_df)
    tmp_data.drop(index = tmp_df.index, inplace = True)
    # 總次數
    a0 = len(tmp_df)
    
    lis_tmp = list(set(tmp_df['total_times']))
#     tmp_data.drop(tmp_data[tmp_data['cano'] == i], inplace = True)
#     print(tmp_data[tmp_data['cano'] == i].index)
#     tmp_data['total_times'].replace()
#     print(tmp_data[tmp_data['cano'] == i])
    for a in lis_tmp:
        
        b = tmp_df[tmp_df['total_times'] ==  a].index
#         print(b.values)
#         print(a,a0)
        tmp_df.loc[b, 'total_times'] = a0
#         tmp_df.loc[b.values,'total_times'] = a0
#         tmp_df.loc[:]['total_times'] = a0
#         tmp_data[tmp_data['cano'] == i].loc[tmp_data['cano'] != a0,'col2'] = a0
#         df.loc[df['col1'] !=' pre','col2']=Nonpre
    print(tmp_df['total_times'])
#     print(lis_tmp)
    # 月平均單日次數
        # set是尋找元素, 在資料集的 locdt也就是日期欄位尋找元素相當於找出在資料集中所有單獨的日子，
        # 也就是資料集中的每個日期 (like 1~30) 
        # len則是知道這個set裡面的元素個數，也就是一個月有幾天
#     a1 = a0/ len(set(data['locdt'])) 
    
#     # 月平均頻率
#     lis_day = list(set(tmp_df['locdt']))
#     for _ in lis_day:
#         # 單日有幾筆
# #         len(tmp_df[tmp_df['locdt'] == _])
#         lis_day.append( len(tmp_df[tmp_df['locdt'] == _]) )
#     for i in 
#     tmp_data[tmp_data['cano'] == i]['total_times'] = a0
#     print(tmp_data[tmp_data['cano'] == i])
#     print(tmp_data[tmp_data['cano'] == i]['total_times'])
#     # 三月單日頻率
#     a2 = a1
    # 第一位的資料是卡號
    lis.append([i, a0])
#     X_data[X_data['']]
print(lis)

# lis[0][1]


# In[ ]:


set(a['locdt'])
len(set(a['locdt']))

lis_locdt=list(set(a['locdt']))
print(lis_locdt)


# In[ ]:


lis_bac_day = []


# In[ ]:



for i in lis_locdt:
#     print(i) # i = 日期
    locdt_i = a[ a['locdt']==i] 
    #print(len(locdt_i)) #當天交易數量
    #lis_bac_day.append(len(locdt_i))
    lis_bac_day.append([i, len(locdt_i)])


# In[ ]:


lis_bac_day


# In[ ]:


a[a['locdt'] == 90].index
y[y.index.isin(a[a['locdt'] == 90].index)]


# In[ ]:


df_bac_loc = a.copy()


# In[ ]:


a.head()


# In[ ]:


df_bac_loc.head()


# In[ ]:


# df_bac_loc = df_bac_loc.pop('locdt')


# In[ ]:


df_bac_loc.loc['acqic']


# In[ ]:





# In[ ]:





# In[ ]:




