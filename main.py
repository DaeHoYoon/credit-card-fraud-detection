#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import sqlalchemy
from dbmodule import uloaddb, dloaddb
from sqlalchemy import create_engine
from dataprep.eda import create_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# %%
host = 'localhost'
id = 'root'
pw = ****
dbname = 'creditcard'
tbname = 'creditcarddb'
df = pd.read_csv(r'.\data\creditcard.csv')
uloaddb(df, id, pw, dbname, tbname, host)
# %%
dloaddb(host, id, pw, dbname, tbname)
# %%
report = create_report(df)
report.save('creditcard_summary.html')
# %%
# 고객이 사기칠 경우와 그렇지 않은 경우의 확률 확인
nofraudrate = np.round((len(df[df.Class==0])/len(df.Class)*100),2)
fraudrate = np.round((len(df[df.Class==1])/len(df.Class)*100),2)
print(nofraudrate)
print(fraudrate) # y값이 매우 불균형한 것을 확인 할 수 있음
# %%
# 불균형한 데이터 시각화
plt.title('Class distribution')
sns.countplot(data=df, x='Class')
# %%
# 연속형 데이터 시각화
cols=['Amount', 'Time']
plt.figure(figsize=(10,10))

nrow = 1
ncol = 2
for i, col in enumerate(cols):
    plt.subplot(nrow, ncol, i+1)
    plt.title('distribution of {0}'.format(col))
    sns.distplot(df[col])
# %%
# 스케일링
from sklearn.preprocessing import RobustScaler

rb_sc = RobustScaler() # 중간값과 사분위값을 이용한 정규화

df['scaled_amount'] = rb_sc.fit_transform(df[['Amount']])
df['scaled_time'] = rb_sc.fit_transform(df[['Time']])

df.drop(['Amount','Time'], axis=1, inplace=True)
df
# %%
Cols = df.columns[:-2].tolist()
Cols.insert(0, 'scaled_time') # 컬럼 위치 재정렬 (0번 위치에 해당컬럼 위치)
Cols.insert(0, 'scaled_amount')
df = df[Cols]
df
# %%
# 데이터 분리
X_data = df.iloc[:,:-1]
y_target = df['Class']

print(X_data.shape)
print(y_target.shape)
# %%
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

for trainidx, testidx in skf.split(X_data, y_target):
    X_train, X_test = X_data.iloc[trainidx], X_data.iloc[testidx]
    y_train, y_test = y_target.iloc[trainidx], y_target.iloc[testidx]

    print(y_test.value_counts()[1]/ len(y_test)) # 동일한 퍼센테이지로 나뉜 것을 확인함
# %%
df = df.sample(frac=1)

fraudDf = df[df['Class']==1]
nofraudDf = df[df['Class']==0][:492]

usampledf = pd.concat([fraudDf, nofraudDf], axis=0)

newdf = usampledf.sample(frac=1)
print(newdf.shape)
# %%
print('Class 0과 1의 비율') # 0과 1의 갯수를 동일하게 맞췄음
print(newdf.Class.value_counts()[0]/newdf.Class.value_counts()[1],':', newdf.Class.value_counts()[1]/newdf.Class.value_counts()[1])
# %%
# 상관관계 그래프 그리기
cor = df.corr()
mask = np.zeros_like(cor, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize=(15,7))
plt.title('original df correlation')
sns.heatmap(cor, cmap='coolwarm_r', mask=mask, linewidths=0.5)
plt.show()

cor1 = newdf.corr()

fig = plt.figure(figsize=(15,7))
plt.title('new df correlation')
sns.heatmap(cor1, cmap='coolwarm_r', mask=mask, linewidths=0.5)
plt.show()
# %%
# 강한 음의 상관관계 = 10, 12, 14, 17
# 강한 양의 상관관계 = 2, 4, 11, 19

colm = ['V10', 'V12', 'V14', 'V17']
colp = ['V2', 'V4', 'V11', 'V19']

plt.figure(figsize=(20,10))
for i, col in enumerate(colm):
    plt.subplot(1, 4, i+1)
    sns.boxplot(data=newdf, x='Class', y=col)

plt.figure(figsize=(20,10))
for i, col in enumerate(colp):
    plt.subplot(1, 4, i+1)
    sns.boxplot(data=newdf, x='Class', y=col)
# %%
# 상관관계 높은 애들의 이상치 제거 (학습시 가중치가 높아질 우려가 있음)

# V14
v14q1 = np.quantile(newdf['V14'].values, 0.25)
v14q3 = np.quantile(newdf['V14'].values, 0.75)
v14IQR = v14q3 - v14q1
v14low = v14q1 - (1.5*v14IQR)
v14up = v14q3 + (1.5*v14IQR)
print('-- v14의 upper, lower --\n', v14up, ',', v14low, '\n\n')

# V12
v12q1 = np.quantile(newdf['V12'].values, 0.25)
v12q3 = np.quantile(newdf['V12'].values, 0.75)
v12IQR = v12q3 - v12q1
v12low = v12q1 - (1.5*v12IQR)
v12up = v12q3 + (1.5*v12IQR)
print('-- v12의 upper, lower --\n', v12up, ',', v12low, '\n\n')

# V10
v10q1 = np.quantile(newdf['V10'].values, 0.25)
v10q3 = np.quantile(newdf['V10'].values, 0.75)
v10IQR = v10q3 - v10q1
v10low = v10q1 - (1.5*v10IQR)
v10up = v10q3 + (1.5*v10IQR)
print('-- v10의 upper, lower --\n', v10up, ',', v10low, '\n\n')

# 이상치 제거
newdf = newdf[(newdf['V14']>=v14low)&(newdf['V14']<=v14up)]
newdf = newdf[(newdf['V12']>=v14low)&(newdf['V12']<=v14up)]
newdf = newdf[(newdf['V10']>=v14low)&(newdf['V10']<=v14up)]

newdf
# %%
# pca
X = newdf.drop(['Class'], axis=1)
y = newdf['Class']

tsne = TSNE(n_components=2, random_state=0)
time0 = time.time()
X_tsne = tsne.fit_transform(X.values)
time1 = time.time()
print('T-SNE 소요시간:', time1-time0)

pca = PCA(n_components=2, random_state=0)
time0 = time.time()
X_pca = pca.fit_transform(X.values)
time1 = time.time()
print('PCA 소요시간:', time1-time0)

tSvd = TruncatedSVD(n_components=2, random_state=0)
time0 = time.time()
X_tSvd = tSvd.fit_transform(X.values)
time1 = time.time()
print('TruncatedSVD 소요시간:', time1-time0)
# %%
# X_tsne
#%%
# 위 모델 시각화
models = [(X_tsne, 'T-SNE'),(X_pca, 'PCA'),(X_tSvd, 'truncatedSVD')]

plt.figure(figsize=(24,6))
for i, model in enumerate(models):
    ax = plt.subplot(1,3,i+1)
    ax.scatter(model[0][:,0], model[0][:,1], c=(y==0), cmap='coolwarm', label = 'No Fraud', linewidths=2)
    ax.scatter(model[0][:,0], model[0][:,1], c=(y==1), cmap='coolwarm', label = 'Fraud', linewidths=2)
    ax.set_title(model[1], fontsize=14)
    ax.legend()
    ax.grid(True)
# %%
# 분류 알고리즘으로 정확도 측정
# 데이터 분리
X_data = newdf.drop(['Class'], axis=1)
y_target = newdf['Class']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.25)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
# %%
# 4개 알고리즘


lg_reg = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
dt_clf = DecisionTreeClassifier()

models = [lg_reg, knn, svc, dt_clf]

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc_sc = np.round(accuracy_score(y_test, pred),4)
    print(f'{model.__class__.__name__}')
    print('예측정확도 :', acc_sc*100,'% \n')
# %%
# cross validation
for i, model in enumerate(models):
    accs = cross_val_score(model, X_train, y_train, cv=5)
    print(f'{model.__class__.__name__}')
    print('교차 검증 정확도 :', np.round(accs.mean()*100, 3),'% \n')
# %%
# 하이퍼 파라미터 튜닝
lg_params = {"penalty" : ['l1', 'l2'],
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
knn_params = {"n_neighbors" : list(range(2,5,1)), 
             "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']}
svc_params = {'C' : [0.5, 0.7, 0.9, 1], 
             'kernel' : ['rbf', 'poly', 'sigmid', 'linear']}
dt_params = {'criterion' : ['entropy', 'gini'],
            'max_depth' : list(range(2,4,1)),
            'min_samples_leaf' : list(range(5,7,1))}

def gridsearch(model, params, X_test):
    gridCV = GridSearchCV(model, param_grid=params, cv =5)
    gridCV.fit(X_train, y_train)
    bestest = gridCV.best_estimator_
    bestpred = bestest.predict(X_test)
    acc_sc = np.round(accuracy_score(y_test, bestpred),3) * 100

    return bestest, acc_sc
# %%
bestlg, bestlgacc = gridsearch(lg_reg, lg_params, X_test)
bestknn, bestknnacc = gridsearch(knn, knn_params, X_test)
bestsvc, bestsvcacc = gridsearch(svc, svc_params, X_test)
bestdt, bestdtacc = gridsearch(dt_clf, dt_params, X_test)
# %%
print('-----' ,'하이퍼 파라미터 튜닝 결과', '-----')
print(f"{bestlg.__class__.__name__}의 최종 accuracy : {bestlgacc:.1f} %")
print(f"{bestknn.__class__.__name__}의 최종 accuracy : {bestknnacc:.1f} %")
print(f"{bestsvc.__class__.__name__}의 최종 accuracy : {bestsvcacc:.1f} %")
print(f"{bestdt.__class__.__name__}의 최종 accuracy : {bestdtacc:.1f} %")
# %%
models = [bestlg, bestknn, bestsvc, bestdt]

def learningCurveDraw(model):
    trainSizes, trainScores, testScores = learning_curve(model, X_data, y_target, cv=10, n_jobs=1, train_sizes = np.linspace(.1, 1.0, 50))
    trainScoresMean = np.mean(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    plt.plot(trainSizes, trainScoresMean, 'o-', color='blue', label='Training score')
    plt.plot(trainSizes, testScoresMean, 'o-', color='red', label='Cross validation score')
    plt.legend(loc='best')

plt.figure(figsize=(20,20))
for i, algo in enumerate(models):
    plt.subplot(2,2,i+1)
    learningCurveDraw(model)
