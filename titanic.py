
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('train.csv')

df = df.dropna(subset=['Embarked'])
# Embarked_mapping = {'S':1, "C":2, "Q":3}
# df['Embarked'] = df['Embarked'].map(Embarked_mapping)
df['Embarked_C'], df['Embarked_Q'], df['Embarked_S'] = pd.get_dummies(df['Embarked'])['C'], pd.get_dummies(df['Embarked'])['Q'], pd.get_dummies(df['Embarked'])['S']


Sex_mapping = {'female':0, "male":1}
df['Sex'] = df['Sex'].map(Sex_mapping)

df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
df_title = df['Title'].tolist()
titles = []
for i in range(len(df_title)):
    if df_title[i] not in titles:
        titles.append(df_title[i])
titles_age_mean = df.groupby('Title')['Age'].median()
titles_age_mean.tolist()
title_uniq = df['Title'].unique().tolist()
title_uniq.sort()
titles_age_mapping = {}
for i in range(len(titles_age_mean)):
    titles_age_mapping[title_uniq[i]]= titles_age_mean[i]
titles_age_mapping


df['Age'] = df['Age'] .fillna(df['Title'].map(titles_age_mapping))
df.isnull().sum()


from sklearn.preprocessing import StandardScaler

bin=[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
df['Age_cut'] = pd.cut(df['Age'],bin, labels=[0,1,2,3,4,5,6,7,8,9])
pd.value_counts(df['Age_cut'])
scaler = StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), age_scale_param)


from sklearn.model_selection import train_test_split

X, y = pd.DataFrame(), pd.DataFrame()
#X['Pclass'], X['Sex'], X['Age_cut'], X['Embarked'] = df['Pclass'], df['Sex'], df['Age_cut'], df['Embarked'] 
X['Fare_scaled'], X['Pclass'], X['Sex'], X['Age_cut'], X['Embarked_C'], X['Embarked_Q'], X['Embarked_S'] = df['Fare_scaled'], df['Pclass'], df['Sex'], df['Age_cut'], df['Embarked_C'], df['Embarked_Q'], df['Embarked_S']
 
y['Survived'] = df['Survived']
X, y = X.values, y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import validation_curve

# param_range = np.arange(1, 80, 2)

# train_scores, test_scores = validation_curve(RandomForestClassifier(), 
#                                              X=X_train, 
#                                              y=y_train, 
#                                              param_name="n_estimators", 
#                                              param_range=param_range,
#                                              cv=2, 
#                                              scoring="accuracy", 
#                                              n_jobs=-1)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='訓練集準確度')
# plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

# plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='訓練集準確度')
# plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

# plt.grid()
# plt.xlabel('Tree Number')
# plt.ylabel('Accuracy')
# plt.ylim(0.7,0.9)
# plt.show()


# from sklearn.decomposition import KernelPCA

# kpca = KernelPCA(n_components=2, kernel='rbf')
# X_kpca = kpca.fit_transform(X)
# X_train_kpca = kpca.fit_transform(X_train)
# X_test_kpca = kpca.fit_transform(X_test)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
pipe_svc = make_pipeline(SVC(random_state=0))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=2,
                  n_jobs=-1)
gs = gs.fit(X, y)
print(gs.best_score_)
print(gs.best_params_)

# forest = RandomForestClassifier(random_state=1)
# PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
# GSRF=GridSearchCV(estimator=forest, param_grid=PRF, scoring='accuracy',cv=2)
# GSRF.fit(X, y)
# print(GSRF.best_score_)
# print(GSRF.best_params_)


import os
os.getcwd()

final_test_df = pd.read_csv('test.csv')
#final_test_df['Embarked'] = final_test_df['Embarked'].map(Embarked_mapping)
final_test_df['Sex'] = final_test_df['Sex'].map(Sex_mapping)
final_test_df['Age'] = final_test_df['Age'] .fillna(df['Title'].map(titles_age_mapping))
final_test_df['Fare'] = final_test_df['Fare'].fillna(df['Fare'].mean())
final_test_df['Age_cut'] = pd.cut(final_test_df['Age'],bin, labels=[0,1,2,3,4,5,6,7,8,9])
final_test_df['Embarked_C'], final_test_df['Embarked_Q'], final_test_df['Embarked_S'] = pd.get_dummies(final_test_df['Embarked'])['C'], pd.get_dummies(final_test_df['Embarked'])['Q'], pd.get_dummies(final_test_df['Embarked'])['S']

age_scale_param = scaler.fit(final_test_df['Age'].values.reshape(-1, 1))
final_test_df['Age_scaled'] = scaler.fit_transform(final_test_df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(final_test_df['Fare'].values.reshape(-1, 1))
final_test_df['Fare_scaled'] = scaler.fit_transform(final_test_df['Fare'].values.reshape(-1, 1), age_scale_param)

final = pd.DataFrame()
#final['Pclass'], final['Sex'], final['Age_cut'], final['Embarked'] = final_test_df['Pclass'], final_test_df['Sex'], final_test_df['Age_cut'], final_test_df['Embarked'] 
final['Fare_scaled'], final['Pclass'],  final['Sex'], final['Age_cut'], final['Embarked_C'], final['Embarked_Q'], final['Embarked_S']  = final_test_df['Fare_scaled'], final_test_df['Pclass'], final_test_df['Sex'], final_test_df['Age_cut'], final_test_df['Embarked_C'], final_test_df['Embarked_Q'], final_test_df['Embarked_S'] 

# final_kpca = kpca.fit_transform(final)
submission = pd.DataFrame()
submission['PassengerId'] = final_test_df['PassengerId']
submission['Survived'] = gs.predict(final)
submission.to_csv('submission.csv', index=False)