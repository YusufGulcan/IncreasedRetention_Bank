# IncreasedRetention_Bank
This project involves the implementation of a machine-learning model to predict customer churn. The original dataset includes 10000 rows and 12 columns.

### Packages and Libraries:
- Pandas
- Numpy
- Sci-kit Learn
- Statsmodels
- Imblearn
- Scipy
- Qbstyles
- Seaborn
- Matplotlib
- Xgboost
- Lgbm
- Catboost

## Exploratory Data Analysis:
This section analyzes the characteristics of the dataset, like the number of null values, data types, distribution types, outliers, and more. 
To display the characteristics, this section also includes Q-Q plots, histograms, boxplots, and countplots generated using Seaborn and Matplotlib libraries.

![image](https://github.com/YusufGulcan/IncreasedRetention_Bank/assets/105684729/5e80a50c-90fa-4fb7-ae78-8b39be61a123)


## Data Cleaning:
In this section, inefficient points of the dataset are treated. For example, data types of columns are converted into optimal types, the distributions are approximated to the normal distribution. 
Also, variables are encoded and scaled to prepare the dataset for modeling. 

![image](https://github.com/YusufGulcan/IncreasedRetention_Bank/assets/105684729/d87e3180-3b2f-4f6b-8e43-d4f4460fa4c7)

## Feature Engineering / Feature Selection

The findings of the exploratory data analysis show that we can create new columns that benefit the performance of the model. For example, 2 new columns are derived from the "credit_score" and "bank_balance". 
To select the best features for the sake of performance, chi2 and VIF tests are used. Although forward and backward selection methods were also applied, a tangible improvement could not be achieved.  

![image](https://github.com/YusufGulcan/IncreasedRetention_Bank/assets/105684729/36e2fc9e-0157-474b-bc80-7ea7cfa9b60f)   ![image](https://github.com/YusufGulcan/IncreasedRetention_Bank/assets/105684729/0dfe6e3c-9268-4c29-8ff8-bfe93e8d6671)  

## Treating the Imbalanced data

Both SMOTE and Random replication techniques are used but the difference in performance was significant between these two methods. SMOTE method performed much better. 
Also, the hyperparameters of the models that give more importance to the minority class were tweaked like "scale_pos_weight" but the overall performance failed to surpass the existing one.

## Modeling
Three models were used in the project; **XGBOOST**, **LGBM**, and **CATBOOST**. These models performed similarly having slight performance differences in different instances but overall the accuracy rate was always above 89%. 
Stacking these three models increased the performance in all metrics(**Accuracy, recall, precision, AUC-ROC, F1**) by about 1%. 

## Evaluation 
The results of the stacked model were outstanding:

**Accuracy Rate = 91%**

**Precision = 88%**

**Recall Rate = 93%**

**ROC-AUC = 91%**

**F1 Score = 90%**

![image](https://github.com/YusufGulcan/IncreasedRetention_Bank/assets/105684729/22270047-dd97-483f-aa8f-96925be06d5a)

















