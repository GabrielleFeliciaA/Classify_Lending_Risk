# Machine Learning Interpretation In Classifying Lending Risk

## Introduction
The data used in classifying lending risk is data taken via the Kaggle website from the following sources: https://www.kaggle.com/datasets/shravankoninti/janatahack-machine-learning-for-banking?select=train_fNxu4vz.csv

The data used has 14 columns with the following information:
- `Loan_ID` : Loan ID number.
- `Loan_Amount_Requested` : Loan amount offered ($).
- `Length_Employed` : How long have you worked?
- `Home_Owner` : Residential ownership status (Mortage, Rent, or Own).
- `Annual_Income` : Annual income.
- `Income_Verified` : Income verification status (Verified or not).
- `Purpose_Of_Loan` : Purpose of borrowing.
- `Debt_To_Income` : Comparison of the amount of monthly debt with total gross income per month. (%)
- `Inquiries_Last_6Mo` : Number of loan checks in the last 6 months.
- `Months_Since_Deliquency` : The number of months after maturity.
- `Number_Open_Accounts` : Number of accounts used.
- `Total_Accounts` : The total number of accounts the borrower has.
- `Gender` : Gender.
- `Interest_Rate` (Target Variable) : Stating the lending risk category, the greater the interest_rate, the greater the risk.

## Exploratory Data Analysis
At this stage several problems and insights were found in the data, namely:
1. The data type for `Loan_Amount_Requested` feature is in object. Therefore, type casting of the feature from the previous data type to int is needed.
2. The minimum value of `Debt_To_Income` is not suppose to be 0. The value 0 indicated that the creditor did not loan even a cent from the debtor. Therefore, the value 0 will be considered as missing values.
3. The `Income_VERIFIED` feature has the values ​​VERIFIED - income and VERIFIED - source of income. These two value categories can be combined into one category called VERIFIED. Therefore, all values ​​in Income_VERIFIED that have the word VERIFIED as a prefix (VERIFIED ...) will be changed to just VERIFIED.
4. The features that have missing values in it are `Length_Employed`, `Home_Owner`, `Annual_Income`, `Debt_To_Income`, and `Months_Since_Deliquency` which has the most missing values count amongst all features.
   
Here is how data cleaning is done to handle missing values:
- The `Months_Since_Delinquency` feature was removed completely because the percentage of missing values ​​was more than 50%.
- The entire `Loan_ID` was dropped because the feature could not be used for analysis or modelling purposes.
- All records of data that has missing values or null value are dropped considering that the count was pretty much.

5. The features `Loan_Amount_Requested`, `Annual_Income`, `Inquiries_Last_6Mo`, `Number_Open_Account`, and `Total_Accounts` have a positive skew distribution which indicated that there are outliers that made the mean value sloped towards the right.
6. `Debt_To_Income` feature has a normal distribution
7. `Number_Open_Accounts` feature has a strong correlation towards `Total_Accounts` feature. Either one of the feature must be drop since `Number_Open_Accounts` could be explained with the feature `Total_Accounts`.

## Data Preprocessing I
In the first data preprocessing stage, data transformation, data encoding, data scaling and data splitting are carried out. The following are the things to do for each stage:
- Data Transformation: The feature `Number_Open_Accounts` is needed to be drop since it has a multi-collinearity proven by the previous correlation check. And, from the distribution check previously, outliers has been detected. Therefore, the features that has outliers in it will type cast to object data typed with each new categories represent the existed range of values.
- Data Encoding: Nominal data type will be encoded with OneHotEncoder method and ordinal data type will be encoded with OrdinalEncoder method.
- Data Scaling: Scaling will be perform for `Debt_To_Income` feature. Since the feature is normally distributed, StandardScaler method will be perform.
- Data Splitting: Independent Data and Dependent Data were divided into train and test data by using Stratify sampling with the intention to keep the proportion of the class division equal. Training data will be divided and have 80% of the original data, while testing data will have the remaining 20% of the original data.

## Modelling
In modelling, the first modelling stage was carried out using logistic regression and random forest classifier. Here are the results:
- When using the whte-box model such as Logistic Regression, the ROC score generated is quite well. However, the model's accuracy is no higher than 50% which indicates that each prediction is not any better than random guessing.
- When using the black-box model such as Random Forest Classifier, the model's accuracy and the ROC score were worse than the model's accuracy and the ROC score of Logistic Regression.
  
From the model's results, Clustering is considered to be used. Clustering might gives insight how many class is suitable using the pattern of the data.

## Clustering
At the clustering stage, the original or previous labels are discarded in order to create new labels. The method being used for clustering is the KMeans algorithm and the yellowbrick library is being used to determine easily the suitable number of classes or K value by visualization. From the visualization of Elbow Method and silhouette score, it is proven that the optimal K value is 4.

## Cluster Analysis
The following is an explanation for each cluster found:
- Cluster 0 has 21346 records of data, cluster 1 has 23567 records of data, cluster 2 has 34902 records of data, and cluster 3 has 32609 records of data.
- Cluster 0 (Low Risk) Debtors that belong to this cluster has the characteristics: has the highest loan rate amongst all clusters, has been employed for more than 5 years, has the highest annual income, oftenly making loans, and has the lowest debt to income ratio.
- Cluster 1 (Low-Medium Risk) Debtors that belong to this cluster has the characteristics: has a pretty low loan rate, has been employed for more than 10 years, has a pretty low annual income, oftenly making loans, and has a pretty high debt to income ratio.
- Cluster 2 (High-Medium Risk) Debtors that belong to this cluster has the characteristics: has the least loan rate amongst all clusters, has been employed for more than 10 years, has the lowest annual income, rarely making loans, and has a pretty low debt to income ratio.
- Cluster 3 (High Risk) Debtors that belong to this cluster has the characteristics: has a pretty high loan rate, has been employed for more than 1 year, has a pretty high annual income, oftenly making loans, and has the highest debt to income ratio.

## Data Preprocessing II
In the second data preprocessing stage, feature selection, correlation checking, data scaling, and data splitting are carried out. The following are the things to do for each stage:
- Feature Selection: By using Decision Tree, it is proven that there are only a few features that play a role in creating clusters such as Total_Accounts, Length_Employed, Annual_Income, Debt_To_Income, Loan_Amount_Requested, etc. Feature dropping was carried out after knowing the features that did not contribute to the cluster classification such as Home_Owner, Verified_Income, Number of_Account_Opensions, Last_6 Months Questions, Loan_Purpose, and Gender.
- Check Correlation: Look at the correlation of each variable using a heatmap, and the results are that all features do not have multicollinearity or the relationship between all independent variables is independent of each other.
- Data Scaling: Scaling the `Loan_Ammount_Requested`, `Annual_Income`, and `Debt_To_Income` features using RobustScaler.
- Data Splitting: Separated dependent and independent variables by dropping the feature cluster and store the values of other features into a variable called x (independent variable), and store the values of feature cluster in a variable called y (dependent variable). Then, the train and test set are divided with the proportion ratio 80:20. Train data will receive 80% of the overall data while test set will receive 20% of the overall data. Lastly, stratify sampling is used for the purpose of sampling so that the class proportions are equally divided and the frequency of each class for test data and train data is equal.

## Modelling II
In the second modeling stage, it was carried out using the logistic regression algorithm, KNN, Adaboost and Gradient Tree Boosting. Here are the results:
- The model using the Logistic Regression algorithm produces an accuracy of about 74% and the average ROC AUC value is 92%. The smallest difference between precision and recall values ​​is about 2% and the maximum is about 22% for all classes. 
- The model using the K-Nearest Neighbors Classifier algorithm produces an accuracy of about 95% and the average ROC AUC value is 99%. The smallest difference between precision and recall values ​​is about 2% and the maximum is about 9% for all classes.
- The model using the AdaBoost Classifier algorithm produces an accuracy of around 87% and the ROC AUC value is 98%. However, the smallest difference between precision and recall values ​​is about 1% and the maximum is about 2% for all classes.
- The model using the Gradient Boosting Classifier algorithm produces an accuracy of around 99% and its ROC AUC value is up to 99.9%. Seeing from the difference in precision and recall values, it can be concluded that the model can predict well. For all classes, the maximum difference between precision and recall is 1%.

## Conclusion
By using the target variable provided from the dataset, the results of the modelling carried out have a poor accuracy value. However, after performing Clustering, it was found that the optimal number of clusters was 4. So, modelling was done again. It was found that the algorithm that has the best performance is the Gradient Tree Boosting algorithm, which has an accuracy value up to 99% and the ROC AUC value is as high as 99.9%. The algorithm has been able to predict the value of each class well, and can be proven through the precision and recall values ​​where the maximum difference between precision and recall values ​​is only 1% for all classes unlike the other models that has the minimum difference between precision and recall values as high as 1% or even more


