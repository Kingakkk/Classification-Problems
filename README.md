# CHURN IN TELCO INDUSTRY

Prediction of churn in Telecom industry - Classification problem

This project is made as a summary of the knowledge gained from the Data Science bootcamp organized by Sages

Project name: "Uczenie maszynowe w przewidywaniu churnu branzy telekomunikacyjnej"

## Loyal vs Churn
The aim of this project is to predict whether a given customer of one of the telecommunications companies will leave or stay. To be able to recognize customers behavior it will be use machine learning based on classification models such as Random Forest, GradientBoosting or XGBoost. 

Dataset is from Kaggle: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset in csv format and each row contain information about one customer. The file with data has been added to this repository.

### Description of variables:
- state: the state the user lives in
- account length: the number of days the user has this account
- area code: the code of the area the user lives in
- phone number: the phone number of the user
- international plan: true if the user has the international plan, otherwise false
- voice mail plan: true if the user has the voice mail plan, otherwise false
- number vmail messages: the number of voice mail messages the user has sent
- total day minutes: total number of minutes the user has been in calls during the day
- total day calls: total number of calls the user has done during the day
- total day charge: total amount of money the user was charged by the Telecom company for calls during the day
- total eve minutes: total number of minutes the user has been in calls during the evening
- total eve calls: total number of calls the user has done during the evening
- total eve charge: total amount of money the user was charged by the Telecom company for calls during the evening
- total night minutes: total number of minutes the user has been in calls during the night
- total night calls: total number of calls the user has done during the night
- total night charge: total amount of money the user was charged by the Telecom company for calls during the night
- total intl minutes: total number of minutes the user has been in international calls
- total intl calls: total number of international calls the user has done
- total intl charge: total amount of money the user was charged by the Telecom company for international calls
- customer service calls: number of customer service calls the user has done
- churn: true if the user terminated the contract, otherwise false

### Describtion about project:
Using differnt machine learning classificator I will try to predict churn of customers. As such disproportions often appear in such data sets (customer churns), it is necessary to investigate whether this does not occur here and, if so, try to apply appropriate methods to solve this problem. In the dataset is 21 variable so maybe it will be usefull to drop some of them or try to change them somehow.

### Steps to follow:
1. Importing needed libraries, setting options, creating functions
2. Importing data
3. Basic information and metrics about dataset
4. Visualization and correlation - first conclusions
5. Feature selection and dimension reduction
6. Classification with different models
7. Crossvalidation to see if model is overfitting or not
8. Improving prediction with best model
9. Summary






