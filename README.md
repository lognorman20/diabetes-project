# Pima Indians Diabetes Project

The point of this project is to determine if someone has diabetes based on data given from the Pima Indians Diabetes Database. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict  whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The dataset consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.


## Project Overview
In this project, I: 

* Created a model with 85% accuracy that predicts if a patient has diabetes based on various health measures to help doctors focus on certain patients.
* Used matplotolib & seaborn to visualize the data and understand it better.
* Engineered features from the data to quantify the value certain health factors such as BMI have on a patient's diagnosis.
* Optimized Gradient Boost Classifier using GridsearchCV to reach the best model. 
* Productionized the model by pickling it into an object that can used in the future.

## Code and Resources Used 
**Python Version:** 3.8 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, pickle, xgboost

## Data Cleaning
After downloading the data from Kaggle, I took a look at it and saw that there were some missing values. These values came in the form of zeros. For example, nobody can have a BMI value of 0. To avoid this issue, I:

*	Dropped rows with a missing glucose value
* Imputed missing insulin values using the median of the insulin column
* Imputed missing blood pressurve values using the median of the blood pressure column
* Imputed missing BMI values using the median of the BMI column
*	Scaled the dataset to standardize the independent features present in the data and prevent machine learning algorithms from weighing greater values higher and consider smaller values as the lower values

## Model Building 

I split the data into train and tests sets with a test size of 20%. 

### Testing Phase

I decided to use this project as a time to test out as many classification models as I could so that I would be able to learn how each one worked. At this time, I did not scale my data, so some of these models might not be as accurate as they could have been. During this testing phase, I used the following models:

* LogisticRegression
* RidgeClassifier
* BaggingClassifier
* GaussianNB
* LinearSVC
* DecisionTreeClassifier
* RandomForestClassifier
* KNeighborsClassifier
* SVC
* XGBClassifier

I used a for loop to gather the accuracy of each model, which probably was not the best idea. Next time, it would make more sense to have a pipeline for each model and run that. To evaluate each model, I decided to use a classification report so that I could see how many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives. After completing this project, I see that the function of these models are quite similar  and yield interchangeable results.

The top performers of this phase were LinearSVC, LogisticRegression, and the RidgeClassifier. They all had an accuracy of 0.69. Each of these models work in a very similar fashion, which helps me understand why all three models had identical f1-scores. These f1-sores were very hit or miss. All of the models had an f1-score of 0.82 for outcomes of '0', but for the outcome '1', their scores were stuck at 0.

### Real Model Building

After this testing phase, I thought that I had gained enough knowledge through research to improve my model selection and development. 

I selected four models:
* Random Forest Classifier
* XGBoost Classifier
* KNN Classifier
* Gradient Boost Classifier

I chose these models because they all had enough variation in their functionality that I thought would be useful due to their evaluation methods for this classification problem. The Random Forest Classifier is, well, random. XGBoost uses a number of nifty tricks like computing second-order gradients, i.e. second partial derivatives of the loss function (similar to Newton’s method). The KNN Classifier finds the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label. And finally, the Gradient Boost Classifier uses the loss function of the base model (e.g. random forest) as a proxy for minimizing the error of the overall model. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest Classifier** : 0.771
*	**XGBoostClassifier**: 0.78
*	**KNeighborsClassifier**: 0.77
*	**GradientBoostingClassifier**: 0.85

The GradientBoostingClassifier far outperformed the other approaches on the test and validation sets. 

## Productionization 
In this step, I pickled my model and saved it into a callable object that can be used on other datasets.

## Code & resources Used

### EDA:
https://github.com/KriAga/Pima-Indians-Diabetes-Dataset-Classification/blob/master/Final.ipynb
https://github.com/KriAga/Pima-Indians-Diabetes-Dataset-Classification/blob/master/EDA%20Diabetes.ipynb
### Data
https://www.kaggle.com/uciml/pima-indians-diabetes-database
### Models
https://www.kaggle.com/omkarsabnis/diabetes-prediction-using-ml-pima-dataset
https://towardsdatascience.com/how-to-train-and-compare-machine-learning-models-with-few-lines-of-code-b1d5e1e266dd
https://github.com/krishnaik06/Diabetes-Prediction/blob/master/Diabetes_Prediction.ipynb
## Acknowledgements
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

