# ML-Projects


### Prediction of churn in Telecom industry
Classification problem - data set from: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset

1.Import data and python packages
- Import packages
- Import data
- Data shape and info

2. EDA (Preprocessing & Data Viz):
- null values is there any null value?
- categorical and numerical data - how many? -> więcej w zbiorze: Bank_loan_modelling Bank_looking_for_clients
- outliers
- corellation
- check if data set is unbalance
- Label Encoding (how to group numerical value in bins -> bank marketing: notebook 1)

EXAMPLE bank modeling (notebook 2)


Label Encoder / get_dummies
TRAIN_TEST_SPLIT:
- Standarization after split and before model
- try models without any modification
- try model with selection best features (many option: for each option built new model and test it)
- try pca with new model (new train test split)

feauture selection -> Breast Cancer (notebook 1)
- check model with crossvalidation using best features and yet without optymize hiperparameters.
- after seleted best features make model from scratch one more time using this choosen features and using grid search CV check hiperparameter for better prediction


2. Data transformation:
- Standarization
- Normalization
- Regularization

3. Variable Selection:
- Select From Model

4. Dimmension reduction
- PCA

5. Unbalance data set:
- SMOOTH
- 

6. Model selection
- Logistic Regression
- SVM
- Decision Tree
- Random Forest / Bagging Classifier
- Gradient Boosting Classifier
- Ada Boost
- XGBoost
- LightGBM
- Naive Bayes
- KNN

7. Validation, checking metrics for different models and comapriing them
- Precision
- Recall
- F1 Score
- Accuracy
- Total 
- Mislabel
- Confussion Matrix
- Roc Auc Score
- Roc Curve
- Gini
- Classification Report

EXAMPLE: bank modeling (notebook 1)

8. Selecting the best model


