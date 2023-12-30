# Heart Stroke Analysis
#
### Overview:
The data is being taken from kaggle website: https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data
#
### Purpose:
The purpose of this project is to leverage machine learning techniques for heart stroke analysis, with the aim of developing a predictive model that can assess and identify potential risks of heart strokes. By analyzing relevant medical data, our goal is to create a tool that provides early detection and risk assessment, empowering healthcare professionals to take proactive measures in preventing heart strokes.
#
### Software Used:
Python 3 (Jupyter Notebook)
#
### Algorithms Used:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
#
### Feature Selection:
1. Knowlege about the data
2. Chi square test
#
### Model Performance:
1. Confusion matrix
2. Accuracy Score
3. AUC and ROC curv
#
### Problems encountered
1. Outliers
2. Multicollinearity
3. Overfitting
#
### Note:
###### 1. In this case 60% of the data is used for training the model and 40% of the data is used to test the model.
###### 2. From the use of confusion matrix and accuracy score, we say that Logistic Regression gives 85.4% accuracy for training data and 84.7% for test data. Whereas, XGBoost gives 98.1% accuracy for training data and 82.6% for test data. Similarly, the accuracy of Random Forest for training data is 99.9% and 84% for test data.  
###### 3. According to AUC and ROC curve Random Forest suits well for training data as it has the highest AUC (i.e., 1) as compaired to other two models (i.e., AUC of Logistic Regression = 0.5187 and AUC of XGBoost = 0.9361). On the other hand, XGBoost suits well for the test data (according to ROC and AUC curve) as it has the highest AUC (i.e., 0.5231) when compaired with the other two models (i.e., AUC of Logistic Regression = 0.5197 and AUC of Random Forest = 0.5177).
###### Thus we conclude that Random Forest algorithm suits better for the training dataset and XgBoost for the test dataset of the considered data.
