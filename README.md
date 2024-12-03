# CKD_is_not_for_me

OVERVIEW

Chronic kidney disease (CKD) is a medical condition that is becoming more common. It reduces the productivity of the kidneys and eventually damages them.​

The objective of the dataset is to diagnostically predict whether a patient is suffering from chronic kidney disease. The dataset includes several diagnostic measurements that are used to make the prediction. The data was collected from a variety of sources, including hospitals, clinics, and doctor's offices. The data is presented in a tabular format, with each row representing a patient and each column representing a diagnostic measurement. The target variable is a binary variable that indicates whether the patient has chronic kidney disease.

The dataset comprises 400 rows of data, each row describing a patient and comprising 25 features such as red blood cell count and white blood cell count. The target variable is "classification," which can have two values, "ckd" or "notckd," where "ckd" stands for chronic kidney disease.​

Linear regression is a statistical method used to model the relationship between a dependent variable (also known as the target variable) and one or more independent variables (also known as features). The goal of linear regression is to find the best-fitting straight line (or hyperplane in higher dimensions) that predicts the dependent variable based on the values of the independent variables.​
​
It is widely used in various fields for predictive modeling and is a foundational concept in statistics and machine learning. ​


Optional : Linear regression is a statistical method that employs a linear equation to model the relationship between a dependent variable and one or more independent variables. The linear equation is a function of the independent variables. For each independent variable, the linear equation computes a linear coefficient that describes the strength of the relationship between the independent variable and the dependent variable. The linear equation is then used to predict the value of the dependent variable for a given set of values of the independent variables.
​

DATASET​
​ The dataset under consideration consists of several medical predictor variables and one target variable, Class. The predictor variables include Blood Pressure (Bp), Albumin (Al), etc. The target variable is either "ckd" or "notckd," where "ckd" denotes chronic kidney disease. The data was collected over a two-month period in India in 2015. The dataset consists of 400 rows and 26 columns. The dataset was created as part of a clinical study of patients with chronic kidney disease

Data Cleaning & Pre Processing​

This dataset is intended for predictive modelling. We encoded categorical variables like 'rbc' into numerical format, which is essential for machine learning algorithms that require numerical input. This allows the model to understand and utilize the information effectively. We also handled missing data by filling in missing values with the mean of that column and replacing NaN with 0, which ensures that the dataset remains intact and usable for further analysis. Finally, we cleaned string data in the 'classification' column by removing leading and trailing whitespace characters. This allows for more effective operations, such as comparisons, merges, or aggregations, without the risk of errors due to unexpected whitespace.

Database Management

We used PostgresSQl design and build a data base to manage and store the Patient data.
We utilized 2 different tables
patient and patient_legend
This structure helps us effectively manage patient data for use in both a predictive model and to maintain data meaning.
The patient data is the stripped down clean numerical only data for input into the predictive model, the patient legend contains data for certain columns in the original dataset where the context of the data was qualitative in nature.  This way we can maintain the original data interpretation whilst also having a clean input into a predictive model.
Any future data would be added to each of the databases (scalability) and would be used to ascertain the spirit of the data while also providing a series of numbers to be crunched in a numerical model.


Machine Learning

import Necessary Libraries​
from sklearn.model_selection import train_test_split​
from sklearn.linear_model import LinearRegression​
from sklearn import metrics​
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report​
​
Load Your Dataset ​
Load your dataset into a pandas DataFrame.​
​
The first step in preparing data for machine learning is to define the target variable (y) and the feature set (x). The target variable is the outcome we want to predict, while the feature set is the input features that we use to make the prediction. 
The second step is to split the dataset into training and testing sets. This ensures that the model is trained on a specific portion of the data (80%) and tested on a separate portion (20%), which is crucial for evaluating its performance.
A logistic regression model is created and trained on the training data. The model learns the relationship between the features and the target variable. 
The model is then used to make predictions on both the training set and the testing set. This allows for an assessment of how well the model performs on known data versus unknown data. 
The accuracy of the model is calculated for both the training and testing sets. 

A confusion matrix is generated for the testing data, which shows the number of true positives, true negatives, false positives, and false negatives. The confusion matrix helps to visualize the model's performance in more detail.

Model Performance:​
The model we created demonstrates a high degree of accuracy in predicting the outcome of a given event. This is evident from the large number of instances that were correctly classified relative to the total number of predictions. In this case, we had 33 true negatives and 46 true positives. In other words, for every instance where the model predicted a negative outcome, the actual outcome was also negative. Similarly, for every instance where the model predicted a positive outcome, the actual outcome was also positive. The model did make a few mistakes, but the number of misclassifications was very small. There was only one false negative, which means that the model incorrectly predicted a negative outcome when the actual outcome was positive. There were no false positives, which means that the model never incorrectly predicted a positive outcome when the actual outcome was negative.

​
Class Distribution:​
The model exhibits a higher capacity for identifying positive instances than negative ones. This is indicated by the high number of true positives (46) compared to the single false negative (1). These results suggest that the model is effective in predicting the positive class, but may not be as effective in predicting the negative class.

Precision:​
For class 0: Precision is 0.97, meaning 97% of the instances predicted as class 0 were correct.​
For class 1: Precision is 1.00, meaning 100% of the instances predicted as class 1 were correct.​
​
Recall:​
This metric indicates the proportion of true positive predictions among all actual positive instances. The model correctly identified all actual instances of class 0. For class 1, the model correctly identified 98% of the actual instances. In this case, the model performed very well for class 0, which is the majority class. For class 1, which is the minority class, the model also performed well, correctly identifying 98% of the actual instances.

​
F1-Score:​
The F1 score is a measure that is used to evaluate a binary classification model. It is the harmonic mean of precision and recall, which are two other measures that are used to evaluate binary classification models. The F1 score is a single metric that balances both precision and recall. It is especially useful when both false positives and false negatives must be accounted for. For both classes, the F1 score is approximately 0.99, which indicates a strong balance between precision and recall.

Accuracy:​
The overall accuracy of the model is computed as the ratio of correctly predicted instances to the total instances. In this case, the accuracy is 0.99, which means that the model correctly predicted 99% of the instances. This is a very good result, indicating that the model is performing well.

Inference​
The classification report for the model shows that it performs exceptionally well, with high precision, recall, and F1-scores for both classes. These are important measures of the model's ability to correctly classify data. However, it is also important to consider the context of the data and the potential consequences of misclassifications. The goal of the analysis should be to ensure that the model's performance is aligned with the goals of the analysis. For example, if the goal is to identify fraudulent transactions, then a high precision score is critical. On the other hand, if the goal is to identify high-risk patients, then a high recall score is more important. In general, the relative importance of precision and recall depends on the context of the analysis.

FOLDER KEY
Root Folder 
CKD.ipynb ---- houses the code used to clean and prep the data before input into machine learning model and SQL database
data_model.ipynb ---- houses code for machine learning script

Database Folder
CKD_ERD --- ERD Diagram for SQL database
CKD_SQL --- Querys for creating database
patient and patient_legend --- CSVs containing the data for input into the database.

Archive Folder
kidney_disease.csv --- original dataset
cleaned_ckd --- dataset for input into machine learning model

.ipynb_checkpoints
contains history of updates.



