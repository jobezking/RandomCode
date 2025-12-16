#Useful data transformation functions
from numpy import log, log1p
from scipy.stats import boxcox
import seaborn as sns
import math

#Log transformation example
#plot histogram and density plot
sns.distplot(data, bins=20)

log_data = [math.log(d) for d in data['Unemployment']]
sns.distplot(log_data, bins=20)

#Polynomial Features: Syntax
#import the class containing the transformation method
from sklearn.preprocessing import PolynomialFeatures
#Create an instance of the class (choose number of degrees or exponents)
polyFeat = PolynomialFeatures(degree=2)
#Create the polynomial features then transform the data
polyFeat  = polyFeat.fit_transform(X_data)
X_poly = polyFeat.transform(X_data)
#Common variable Transformations
#feature type: continuous. Transformation: Standard, min-max, robust scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#nominal: categorical unordered features (True/False): binary, one-hot encoding (0,1)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from pandas import get_dummies
#ordinal: categorical, ordered features (movie ratings): Ordinal Encoding (0, 1, 2, 3)
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OrdinalEncoder

#Examining churn data, churn value by payment type
sns.barplot(y='churn_value',x='payment',data=df_phone, ci=None)
plt.ylabel('Churn Rate')
plt.xlabel('Payment Type')
plt.title('Churn Rate by Payment Type, Phone Customers')

#Examining churn data by tenure
sns.barplot(y='churn_value',x=pd.cut(df_phone.months,bins=5),data=df_phone, ci=None)
plt.ylabel('Churn Rate')
plt.xlabel('Tenure Range in Months')
plt.title('Churn Rate by Tenure, Phone Customers')

#Pair plot; Seaborn plot, feature correlations
pairplot = df_phone[['months','gb_mon', 'total_revenue', 'cltv', 'churn_value']]
sns.pairplot(pairplot, hue='churn_value')

#Seaborn hexibin plot
sns.jointplot(x=df[labels['months']], y=df[labels['monthly']],kind='hex')