# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Importing all datasets
df = pd.read_csv("C:\\Users\\sumne\\Downloads\\Lead+Scoring+Case+Study\\Lead Scoring Assignment\\Leads.csv")
df.head()

df.shape

df.describe()

# 1.Data Cleaning and Processing

# Dropping irrelevant columns (e.g., Prospect ID, Lead Number)
df.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)
df.head()

#Checking value counts and types to further drop unnecessary columns
for i in df.columns:
    print(df[i].value_counts())

# Column 'Magazine', 'I agree to pay the amount through cheque','Receive More Updates About Our Courses', 'Update me on Supply Chain Content' 
# 'Get updates on DM Content' only have NO values hence are of no use
col_tobe_dropped = ['Magazine', 'I agree to pay the amount through cheque','Receive More Updates About Our Courses', 'Update me on Supply Chain Content','Get updates on DM Content']


df.drop(col_tobe_dropped, axis=1, inplace=True)
df.head()

### Dropping columns with data imbalance

# Checking imbalance, since many columns have highly imbalanced data
df['Do Not Call'].value_counts()

df['What matters most to you in choosing a course'].value_counts()

df.drop(['Do Not Call','What matters most to you in choosing a course'], axis=1, inplace=True)
df.head()

# Handling 'Select' values in categorical variables
df.replace('Select', np.nan, inplace=True)

### NULL Value treatment

#Checking % of null values
(df.isnull().sum()*100)/df.shape[0]

### Columns like 'How did you hear about X Education','Lead Quality','Lead Profile','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Profile Score','Asymmetrique Activity Score' have more than 45% null values

col_tobe_dropped2 = ['How did you hear about X Education','Lead Quality','Lead Profile','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Profile Score','Asymmetrique Activity Score']

df.drop(col_tobe_dropped2, axis=1, inplace=True)
df.head()

### 1.Lead Source

print(df['Lead Source'].value_counts())
print(df['Lead Source'].isnull().sum())

df['Lead Source'].fillna(df['Lead Source'].mode()[0], inplace=True)

print(df['Lead Source'].value_counts())
print(df['Lead Source'].isnull().sum())

### 2.TotalVisits

print(df['TotalVisits'].value_counts())
print(df['TotalVisits'].isnull().sum())

#Let's fill the null values with mode
df['TotalVisits'].fillna(df['TotalVisits'].mode()[0], inplace=True)

#We have three outliers which we can safely remove: 115, 141, 251
df = df[df['TotalVisits']<100]

print(df['TotalVisits'].value_counts())
print(df['TotalVisits'].isnull().sum())

### 3.Page Views Per Visit

print(df['Page Views Per Visit'].value_counts())

print(df['Page Views Per Visit'].isnull().sum())

#Let's fill the null values with mode
df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].mode()[0], inplace=True)

print(df['Page Views Per Visit'].value_counts())

print(df['Page Views Per Visit'].isnull().sum())

### 3.Last Activity

print(df['Last Activity'].value_counts())
print(df['Last Activity'].isnull().sum())

#### Last Activity seems to be descriptive information and is of no use and can be dropped

df.drop('Last Activity', axis=1, inplace=True)


#### Last Notable Activity is also a similar variable and can be dropped
df.drop('Last Notable Activity', axis=1, inplace=True)

### 4.Country

print(df['Country'].value_counts())
print(df['Country'].isnull().sum())

### A significant amount of rows don't have a country. The dataset doesn't talk about any country specific tailored plan so it van be dropped.


df.drop('Country', axis=1, inplace=True)

### 5. Specialization

print(df['Specialization'].value_counts())
print(df['Specialization'].isnull().sum())

#### Since specialization can be an important factor, dropping it won't make sense. Let's replace it with 'Not Specified'

df['Specialization'].fillna('Not Specified', inplace = True)

print(df['Specialization'].value_counts())
print(df['Specialization'].isnull().sum())

### 6.What is your current occupation

print(df['What is your current occupation'].value_counts())
print(df['What is your current occupation'].isnull().sum())

#### Occupation can also be a model building factor, hence it's better to create a new category than drop it. Using mode will create data imblanace 

df['What is your current occupation'].fillna('Not Specified', inplace = True)

print(df['What is your current occupation'].value_counts())
print(df['What is your current occupation'].isnull().sum())

### 7. tags

print(df['Tags'].value_counts())
print(df['Tags'].isnull().sum())

#### Tags more or less, appear to be Sales teams' notes and hence can be dropped

df.drop('Tags', axis=1, inplace=True)

### 8. City

print(df['City'].value_counts())
print(df['City'].isnull().sum())

#### Since this column has scattered values in different categories, It's better to divide all cities in three categories: 'Metro Cities', 'Non-Metro' and 'Not Mentioned'.

def city_categorization(city):
    if city in ['Mumbai', 'Other Metro Cities']:
        return 'Metro City'
    elif city in ['Tier II Cities', 'Thane & Outskirts', 'Other Cities', 'Other Cities of Maharashtra']:
        return 'Non-Metro City'
    else:
        return 'Not Mentioned'

# Let's simplifiy City into three broad City Categories
df['City Category'] = df['City'].apply(city_categorization)

#Let's now drop 'City'
df.drop('City', axis=1, inplace=True)

print(df['City Category'].value_counts())
print(df['City Category'].isnull().sum())

#Check if any further null values
(df.isnull().sum()*100)/df.shape[0]



### Now we have no more null values &#128516;

# 2. Data Transformation

### Label Encoding all Yes/No to 1/0 

label_encod = ['Do Not Email','Search','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','A free copy of Mastering The Interview']

#Replacing all Yes/No to 1/0
df[label_encod] = df[label_encod].replace({'Yes': 1, 'No': 0})

df.info()

### Data types of all the columns seem alright. 
### Exploratory data Analysis: Let's plot all categorical columns to see their distributions and get some basic insights

### Categorical Variables : Univariate Analysis

# Filtering out all the categorical columns
categorical_columns = df.select_dtypes(include = ['object']).columns

# Set up the figure size
plt.figure(figsize=(12, 24), dpi=100)

# Loop through each column and create subplots
for index, column in enumerate(categorical_columns, start=1):
    plt.subplot(4, 2, index)
    (df[column].value_counts(normalize=True)*100).plot.bar(width=0.5, color='maroon')
    plt.title(column)
    plt.xlabel('')
    plt.ylabel('Share')
    plt.xticks(rotation=90)

# Adjust layout and display the combined subplot
plt.tight_layout()
plt.show()


#### Basic Insights: 
- Landing page submission and APIs are responsible for more than 85% of the leads origin.
- Google, Direct Traffic, Olark Chat and Organic search are top four lead sources accounting for around 80% of the total
- Finance, HRM and Marketing managment are top searches of these kinda courses
- Around 60% of the leads came from unemployed professionls, indicating a trend for upskilling to find better prospects
- From the provided information, more leads came from Metro cities

#### Plotting all the Y/N variables.
#### Yes - 1 and No - 0

# Set up the figure size
plt.figure(figsize=(10, 18), dpi=100)

# Loop through each column and create subplots
for index, column in enumerate(label_encod, start=1):
    plt.subplot(4, 2, index)
    (df[column].value_counts(normalize=True)*100).plot.pie(autopct='%1.1f%%', labeldistance=None)
    plt.title(column)
    plt.xlabel('')
    plt.ylabel('Share in %')
    plt.xticks(rotation=90)
    plt.legend()

# Adjust layout and display the combined subplot
plt.tight_layout()
plt.show()


#### Basic Insights:
- Only around 8% of the customers opted for 'Do not email' indicating EMAIL as a preferred method of contact
- Around 31% of the leads marked yes for receiving 'A free copy of mastering the interview', indicating their primary motive of landing a good job. This is also proven by the fact that majority of the leads were given by unemployed people.

numerical_columns = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']

# Set up the figure size
plt.figure(figsize=(12, 4), dpi=100)

# Create box plots for each numerical column

for index, column in enumerate(numerical_columns, start=1):
    plt.subplot(1, 3, index)
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.xlabel(column)


# Adjust layout and display the combined subplot
plt.tight_layout()
plt.show()


### Dealing with Outliers:

#### 1. TotalVisits

df['TotalVisits'].value_counts()

#### Let's fix the upper limit of TotalVisits column to 30, since these outliers(7 in total) can skew the model

# Lets filter the dataset
df= df[df['TotalVisits']<30]

### 2. Page Views per Visit

df['Page Views Per Visit'].value_counts()

#### Binning the 'Page Views Per Visit' variable might be a reasonable approach, especially if you're interested in grouping similar values together and reducing the impact of extreme values. Let's use pd.cut to check the distribution

# Define bin edges
bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, float('inf')] 

# Define bin labels
bin_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-20', '20+']

# Create bins using pd.cut()
df['Page Views Range Per Visit'] = pd.cut(df['Page Views Per Visit'], bins=bin_edges, labels=bin_labels, right=False)


# Let's drop Page views per visit
df.drop('Page Views Per Visit', axis=1, inplace=True)

(df['Page Views Range Per Visit'].value_counts(normalize=True)*100).plot.bar(width=0.5, color='maroon')
plt.title('Page Views Range Per Visit')
plt.ylabel('Share of views in %')
ticks = np.arange(0, 25, 2)
plt.xticks(rotation=0)
plt.yticks(ticks)
plt.show()

sns.pairplot(df[['TotalVisits','Total Time Spent on Website']])
plt.show()

# Creating a Heatmap
correlation_matrix = df[['TotalVisits','Total Time Spent on Website']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

### Basic Insights:
- Total time spent on website vs page views per visit: There appears to be a positive correlation between total time spent on website and page views per visit. This means that people who spend more time on the website tend to visit more pages. This relationship is not perfectly linear, and there is some scatter in the data.

df.info()

# 3. Data Preparation

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True, dtype=int)
df.head()

df.describe()

df.shape

from sklearn.model_selection import train_test_split

# Splitting the dataset into features and target variable
X = df.drop('Converted', axis=1)
y = df['Converted']

X.head()

y.head()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)

# 4. Feature Scaling

# Feature Scaling
scaler = StandardScaler() # Creating an object

X_train[['TotalVisits','Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website']])

X_train.head()

df.head()

### Checking Conversion Rate

conversion = (sum(df['Converted'])/len(df['Converted'].index))*100
conversion

# 5. Exploring Correlations

plt.figure(figsize = (20,10))        # Size of the figure

# Creating a heatmap
sns.heatmap(df.corr(),annot = True, cmap='coolwarm')
plt.show()

# Since heatmap is too cluttery
df.corr()

# 6. Building the Model

# Since Corr map was too cluttery, we can use Manual plus RFE to select features for us
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

logreg = LogisticRegression()

# Starting RFE with 15 features
rfe = RFE(estimator=logreg, n_features_to_select=15)
# Fitting the variable
rfe = rfe.fit(X_train, y_train)

rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

#Creating a list of the selected columns by RFE
col = X_train.columns[rfe.support_]
col

# Checking features removed
X_train.columns[~rfe.support_]

# creating a new dataframe with RFE suggested Features
X_train_RFE = X_train[col]
X_train_RFE.head()

### Creating the first model: Model 1

# Creating new dataframe for newly selected features

#Adding constant as needed while using statsmodels
X_train_1 = sm.add_constant(X_train_RFE)

#Creating and Fitting the model
logm1 = sm.GLM(y_train,X_train_1, family = sm.families.Binomial()).fit()
logm1.summary()

#### 'What is your current occupation_Housewife' has a very high -p value

#calculating VIF for Model 1
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_RFE[col].columns
vif['VIF'] = [variance_inflation_factor(X_train_RFE[col].values, i) for i in range(X_train_RFE[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#### All the VIF values are way below 5 and are a good indicator. But let's try another model without 'What is your current occupation_Housewife'

### Creating the second model: Model 2

# Removing 'What is your current occupation_Housewife' and previously added 'const'
X_train_rfe_2 = X_train_1.drop(['What is your current occupation_Housewife','const'],axis = 1)
X_train_rfe_2.head()

# Creating and fitting the second model

X_train_2 = sm.add_constant(X_train_rfe_2)  


logm2 = sm.GLM(y_train,X_train_2,family=sm.families.Binomial()).fit()
logm2.summary()                                                         

# Calculating VIF again
vif2=pd.DataFrame()
vif2['Features']= X_train_rfe_2.columns  
vif2['VIF']=[variance_inflation_factor(X_train_rfe_2.values,i) for i in range(X_train_rfe_2.shape[1])]
vif2['VIF']=round(vif2['VIF'],2)
vif2=vif2.sort_values(by='VIF',ascending=False)
vif2

#### All the VIF values are way below 2 and p-values are below 0.05. We have a winner i.e Model 2

# 7. Making Predictions

y_train_pred = logm2.predict(X_train_2)

# Creating a new dataset to predicted values in it
y_train_pred_final = pd.DataFrame({'Converted':y_train.values,'Convert_prob':y_train_pred,'ID':y_train.index})
y_train_pred_final.head() 

# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Convert_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()

from sklearn import metrics
# Let's check the overall accuracy.
overall_accuracy_logm2 = metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
overall_accuracy_logm2

# Let's take a look at the confusion matrix
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion

# 8. Step 9: Plotting the ROC Curve
An ROC curve demonstrates several things:

It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Convert_prob, drop_intermediate = False )

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Convert_prob)

#### Closer the ROC curve to the left border, the better it is

# 9.: Finding Optimal Cutoff Point
Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Convert_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]
print(cutoff_df)

# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensitivity','specificity'])
plt.show()

### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# Let's predict the outcome with this cutoff of 0.3
y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()

# Let's check the overall accuracy now 
Accuracy_train = metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
Accuracy_train

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
Sensitivity_train = (TP / float(TP+FN))*100
Sensitivity_train

Specificity_train = (TN / float(TN+FP))*100
Specificity_train

# Calculate false postive rate - predicting Converted when customer hasn't converted
print((FP/ float(TN+FP))*100)

# Positive predictive value 
print((TP / float(TP+FP))*100)

# Negative predictive value
print ((TN / float(TN+ FN))*100)

## Precision and Recall

Precision
TP / TP + FP

from sklearn.metrics import precision_score, recall_score
Precision_Train = (confusion2[1,1]/(confusion2[0,1]+confusion2[1,1])*100)
Precision_Train

Recall
TP / TP + FN

Recall_Train = (confusion2[1,1]/(confusion2[1,0]+confusion2[1,1])*100)
Recall_Train

#### High recall indicates that the model is capturing a large proportion of actual positive instances, which is desirable in scenarios where it's important to avoid false negatives which in this case is a probable lead

### Precision and recall tradeoff - train set

from sklearn.metrics import precision_recall_curve

y_train_pred_final.Converted, y_train_pred_final.predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Convert_prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

# 10. Making predictions on the test set

# Scaling the variables of the test set. Using only transform and not fit

X_test[['TotalVisits','Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website']])

X_test.head()

# Using the same list of columns as in the model 2(logm2)
col_test = X_train_2.columns[1:] 

# Amending final set accordingly
X_test_final = X_test[col_test]               

# Adding constant as needed while using statsmodel
X_test_final = sm.add_constant(X_test_final)

y_pred_test = logm2.predict(X_test_final)      # Predicting the final test set

### Making predictions on the test set

# Saving the predictions in a new dataset

y_test_pred_final = pd.DataFrame({'Converted':y_test.values,'Converted_prob':y_pred_test,'ID':y_test.index})

y_test_pred_final.head()

y_test_pred_final[:10]

# Model Evaluation

# Predicting the outcomes with probability cutoff as 0.3 by creating new columns in the final test dataset

y_test_pred_final['Predicted']=y_test_pred_final['Converted_prob'].map(lambda x:1 if x >0.3 else 0 )  # Predicted value 

y_test_pred_final.head()

# Creating confusion matrix for test pred set
confusion_test=confusion_matrix(y_test_pred_final.Converted,y_test_pred_final.Predicted)
confusion_test

#Sensitivity score of test set
Sensitivity_test =(confusion_test[1,1]/(confusion_test[1,0]+confusion_test[1,1])*100)  
Sensitivity_test

#specificity score of the test set
Specificity_test =(confusion_test[0,0]/(confusion_test[0,0]+confusion_test[0,1])*100)  
Specificity_test

## Precision And Recall of the test set

#  Pecision score
Precision_test = (confusion_test[1,1]/(confusion_test[0,1]+confusion_test[1,1])*100)
Precision_test

#  Recall score
Recall_test = (confusion_test[1,1]/(confusion_test[1,0]+confusion_test[1,1])*100)
Recall_test

# Accuracy score of the test set

Accuracy_test = metrics.accuracy_score(y_test_pred_final.Converted,y_test_pred_final.Predicted)*100
Accuracy_test

### Overall Accuracy for test set is > 75% 

from sklearn.metrics import f1_score
print('F1_Score: ',f1_score(y_test_pred_final.Converted, y_test_pred_final.Predicted)*100)

#### The F1-score is a metric that combines both precision and recall into a single value, providing a balanced assessment of a classification model's performance. It is particularly useful when there is an uneven class distribution, as it considers both false positives and false negatives.
#### A an F1-score of approximately 73. \3\ t suggests thatheur model achieves a good balance between precision and recall. A higher F1-score indicates better overall performance of the model in terms of both false positives and false negatives.

print('Comparing basic metrics of Test and Train Models:')
print()
print('Sensitivity_Train:',Sensitivity_train)   
print('Specificity_Train:',Specificity_train)
print('Precision_Train:',Precision_Train) 
print('Recall_Train:',Recall_Train)
print('Accuracy_Train:',Accuracy_train)
print()
print('-----------------------')
print()
print('Sensitivity_Test:',Sensitivity_test)   
print('Specificity_Test:',Specificity_test)
print('Precision_Test:',Precision_test) 
print('Recall_Test:',Recall_test)
print('Accuracy_Test:',Accuracy_test)

### All scores compared to each other are within the range of 5% which is ideal

# FINAL STEP: Assigning a lead score

#Adding the column of lead score

y_test_pred_final['Lead Score']=y_test_pred_final['Converted_prob'].apply(lambda x:round(x*100))

y_test_pred_final.head(50)

