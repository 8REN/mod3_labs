# Telecom Service Termination 


  
  ## Binary Classification on Churn Dataset
  
  
 Author: Brennan Mathis
 
 The contents of this repository contains an analysis of telecommunications 
 data regarding phone usage and customer retention.
 
 data obtained from https://www.kaggle.com/becksddf/churn-in-telecoms-dataset
 
data includes the following info:
       'state', 'account length', 'area code', 'phone number',
       'international plan', 'voice mail plan', 'number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'customer service calls', 'churn'. (telecomchurndataset.csv)
 
 ### Problem:
 
What features are involved in customer retention?
Can customer churn be predicted with data provided?
Are false positives preferred to false negatives?
 
### Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from statsmodels.stats import diagnostic
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import log_loss
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve

from sklearn.pipeline import Pipeline

## perform gridsearch of multiple algorithms to find
## best fit for the job

columns not factored into assessment include: 
'phone number',
'area code', 
'number vmail messages', 
'total day minutes',
'total eve minutes',
'total night minutes',
'total intl minutes',
this is due to high correlation with other features, or irrelevance to assessment

### models

   * RandomForestClassifier(),
   * KNeighborsClassifier(),
   * LogisticRegression(),
   * SGDClassifier(),
   * DecisionTreeClassifier(),
   * SVC(),
   * XGBClassifier(),
   * AdaBoostClassifier(),
   
 ![corrheatmap](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/corrheatmap.png)
 
  ![pair](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/pairplot.png)

 ![statebar](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/statevschurn.png)
 
 ![pie](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/pie.png)
 
 ![comp](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/roccomp.png)
 
 ![comp2](https://github.com/br3nnan8/mod3-classificationproject/blob/master/visualaids/prcurvecomp.png)
 

 
 To expand on this project, I would continue to work on my parameter assessment, create possible 
 ensemble pipeline to yield better results with fewer false negatives
 
 
