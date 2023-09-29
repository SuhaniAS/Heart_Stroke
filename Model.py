#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xg
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[2]:


#import the data
heart = pd.read_csv(r"D:\Suhani\CHD_preprocessed.csv")
heart


# In[3]:


heart.info()


# In[4]:


heart.isnull().sum()


# In[5]:


heart.describe()


# In[6]:


plt.figure(figsize=(13,13))
sns.heatmap(heart.corr(),cmap='Blues',annot=True)
plt.show()


# In[7]:


vif_data=pd.DataFrame()
vif_data["features"]=heart.columns

vif_data["VIF"] = [variance_inflation_factor(heart.values, i)
                          for i in range(len(heart.columns))]
  
print(vif_data)


# In[8]:


contingency_table=pd.crosstab(heart["currentSmoker"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[9]:


contingency_table=pd.crosstab(heart["gender"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[10]:


contingency_table=pd.crosstab(heart["education"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[11]:


contingency_table=pd.crosstab(heart["sysBP"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[12]:


contingency_table=pd.crosstab(heart["diaBP"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[13]:


contingency_table=pd.crosstab(heart["age"],heart["HeartStroke"])
print('contingency_table :-\n',contingency_table)#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)
no_of_rows=len(contingency_table.iloc[0:,0])
no_of_columns=len(contingency_table.iloc[0,0:])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05 
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[14]:


X = heart.drop(["HeartStroke","education","currentSmoker",'glucose','sysBP','diaBP'],axis=1)
y = heart["HeartStroke"]


# In[15]:


X.columns


# In[16]:


#Outlier Detection using Boxplot
import matplotlib.pyplot as plt
data = heart
plt.figure(figsize=(15,15))

plt.subplot(5,2,1)
plt.boxplot(data["age"])
plt.title("age")

plt.subplot(5,2,2)
plt.boxplot(data["cigsPerDay"])
plt.title("cigsPerDay")

plt.subplot(5,2,3)
plt.boxplot(data["BPMeds"])
plt.title("BPMeds")

plt.subplot(5,2,4)
plt.boxplot(data["prevalentStroke"])
plt.title("prevalentStroke")

plt.subplot(5,2,5)
plt.boxplot(data["prevalentHyp"])
plt.title("prevalentHyp")

plt.subplot(5,2,6)
plt.boxplot(data["diabetes"])
plt.title("diabetes")

plt.subplot(5,2,7)
plt.boxplot(data["totChol"])
plt.title("totChol")

plt.subplot(5,2,8)
plt.boxplot(data["BMI"])
plt.title("BMI")

plt.subplot(5,2,9)
plt.boxplot(data["gender"])
plt.title("gender")

plt.subplot(5,2,10)
plt.boxplot(data["heartRate"])
plt.title("heartRate")

plt.show()


# In[17]:


std = StandardScaler()
X = std.fit_transform(X)


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=21,test_size=0.15)


# In[19]:


#fit the Logistic Regression model
model1=LogisticRegression()
model1.fit(X_train,y_train)

y_train_pred_log=model1.predict(X_train)


# In[20]:


cm = confusion_matrix(y_train,y_train_pred_log)
conf_matrix = pd.DataFrame(cm, columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,fmt='d',cmap='Blues',annot=True)
plt.show()

print('The details of Confusion_matrix is:')
print(classification_report(y_train,y_train_pred_log))

print('Accuracy of Logistic Regression Model:',accuracy_score(y_train_pred_log,y_train))


# In[21]:


model2=xg.XGBClassifier()
model2.fit(X_train,y_train)

y_train_pred_xg=model2.predict(X_train)


# In[22]:


cm = confusion_matrix(y_train,y_train_pred_xg)
conf_matrix=pd.DataFrame(data=cm, columns=["Predict:0","Predict:1"],index=["Actual:0","Actual:1"])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Reds')
plt.show()

print("The details of Confusion matrix:")
print(classification_report(y_train,y_train_pred_xg))

print("Accuracy Score of XGBoost:",accuracy_score(y_train_pred_xg,y_train))


# In[23]:


y_test_pred_log=model1.predict(X_test)


# In[24]:


cm = confusion_matrix(y_test,y_test_pred_log)
conf_matrix = pd.DataFrame(cm, columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,fmt='d',cmap='Blues',annot=True)
plt.show()

print('The details of Confusion_matrix is:')
print(classification_report(y_test,y_test_pred_log))

print('Accuracy of Logistic Regression Model:',accuracy_score(y_test_pred_log,y_test))


# In[25]:


y_test_pred_xg = model2.predict(X_test)


# In[26]:


cm = confusion_matrix(y_test,y_test_pred_xg)
conf_matrix=pd.DataFrame(data=cm, columns=["Predict:0","Predict:1"],index=["Actual:0","Actual:1"])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Reds')
plt.show()

print("The details of Confusion matrix:")
print(classification_report(y_test,y_test_pred_xg))

print("Accuracy Score of XGBoost:",accuracy_score(y_test_pred_xg,y_test))


# In[27]:


cross_tab_1=pd.crosstab(heart['BPMeds'],heart['HeartStroke'])
cross_tab_1


# In[28]:


cross_tab_2=pd.crosstab(heart['prevalentStroke'],heart['HeartStroke'])
cross_tab_2


# In[29]:


cross_tab_3=pd.crosstab(heart['prevalentHyp'],heart['HeartStroke'])
cross_tab_3


# In[30]:


cross_tab_4=pd.crosstab(heart['diabetes'],heart['HeartStroke'])
cross_tab_4


# In[31]:


model3 = RandomForestClassifier()
model3.fit(X_train,y_train)

y_train_pred_rf = model3.predict(X_train)


# In[32]:


cm = confusion_matrix(y_train,y_train_pred_rf)
conf_matrix=pd.DataFrame(data=cm, columns=["Predict:0","Predict:1"],index=["Actual:0","Actual:1"])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Reds')
plt.show()

print("The details of Confusion matrix:")
print(classification_report(y_train,y_train_pred_rf))

print("Accuracy Score of XGBoost:",accuracy_score(y_train_pred_rf,y_train))


# In[33]:


y_test_pred_rf = model3.predict(X_test)


# In[34]:


cm = confusion_matrix(y_test,y_test_pred_rf)
conf_matrix=pd.DataFrame(data=cm, columns=["Predict:0","Predict:1"],index=["Actual:0","Actual:1"])

plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Reds')
plt.show()

print("The details of Confusion matrix:")
print(classification_report(y_test,y_test_pred_rf))

print("Accuracy Score of XGBoost:",accuracy_score(y_test_pred_rf,y_test))


# In[35]:


#Plor an roc curve
fpr, tpr, _ = metrics.roc_curve(y_train,y_train_pred_log)
auc = round(metrics.roc_auc_score(y_train,y_train_pred_log), 4)
plt.plot(fpr,tpr,label='Logistic Regression, AUC='+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_train,y_train_pred_xg)
auc = round(metrics.roc_auc_score(y_train,y_train_pred_xg),4)
plt.plot(fpr,tpr,label='XGBoost, AUC='+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_train,y_train_pred_rf)
auc = round(metrics.roc_auc_score(y_train,y_train_pred_rf),4)
plt.plot(fpr,tpr,label='RandomForest, AUC='+str(auc))

plt.legend()
plt.show()


# In[36]:


#Plor an roc curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_test_pred_log)
auc = round(metrics.roc_auc_score(y_test,y_test_pred_log), 4)
plt.plot(fpr,tpr,label='Logistic Regression, AUC='+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_test,y_test_pred_xg)
auc = round(metrics.roc_auc_score(y_test,y_test_pred_xg),4)
plt.plot(fpr,tpr,label='XGBoost, AUC='+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_test,y_test_pred_rf)
auc = round(metrics.roc_auc_score(y_test,y_test_pred_rf),4)
plt.plot(fpr,tpr,label='RandomForest, AUC='+str(auc))

plt.legend()
plt.show()

