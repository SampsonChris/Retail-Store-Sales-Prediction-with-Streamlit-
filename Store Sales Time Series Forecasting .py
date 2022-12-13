#!/usr/bin/env python
# coding: utf-8

# # Goal
# ### To predict store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer and to build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores.
# 
# 
# # Questions
# ### 1. Did promotions influence sales in some cities than the others?
# ### 2. Which date/year records the highest / lowest sales ? 
# ### 3. Did earthquake impact sales in any way?
# ### 4. Which holidays affect sales the most? 
# ### 5. Are sales affected by oil prices and promotions?
# ### 6.Which items have the highest sales /lowest sales 
# 
# 
# # Hypothesis
# 
# ### 1. Promotions Positively affect store sales
# ### 2. Depending on location, sales differ from city to city 
# 

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().system('pip install seaborn ')
import seaborn as sns
get_ipython().system('pip install plotly==4.14.3')
import plotly.graph_objects as go
import warnings
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px 


# # DATA CLEANING

# In[2]:


# Loading train and test datasets 
train_set = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/train.csv')
test_set = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/test.csv')


# In[3]:


#Printing a summary of the train dataset to find errors and missing values.
train_set.info()


# In[4]:


# Previewing train data set dataset 
train_set.head()


# In[5]:


#Printing a summary of the train dataset to find errors and missing values.
test_set.info()


# In[6]:


#Previewing train data set dataset 
test_set.head()


# In[7]:


# Loading supplementary datasets
oil = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/oil.csv')
holidays = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/holidays_events.csv')
stores = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/stores.csv')
transactions = pd.read_csv('/Users/Admin/Desktop/store-sales-time-series-forecasting/transactions.csv')


# In[8]:


#Previewing oil dataset 
oil.head()


# In[9]:


#Printing a summary of the oil dataset to find errors and missing values.
oil.info()


# In[10]:


#filling in missing values 
oil = oil.ffill().bfill()
oil


# In[11]:


#previewing holidays dataset
holidays.head()


# In[12]:


#Printing a summary of the holidays dataset to find errors and missing values.
holidays.info()


# In[13]:


#Previewing stores data set 
stores.head()


# In[14]:


#Printing a summary of the stores dataset to find errors and missing values
stores.info()


# In[15]:


# Previewing transactions data
transactions.head()


# In[16]:


#Printing a summary of the transactions dataset to find errors and missing values
transactions.info()


# In[17]:


transactions.drop_duplicates(inplace = True)
transactions.info()


# In[18]:


stores.drop_duplicates(inplace = True)
stores.info()


# In[19]:


train_set.store_nbr.unique()


# In[20]:


test_set.store_nbr.unique()


# In[21]:


stores.city.unique()


# In[22]:


stores.city


# In[23]:


# checking the range of dates 
train_set.date.min(), train_set.date.max()


# In[24]:


# Adding the sales date column to the train dataset 
train_set['Sales_date'] = pd.to_datetime(train_set['date']).dt.date


# In[25]:


# Checking for unique values in the train data set
train_set.Sales_date.nunique()


# In[26]:


#checking the range of dates 
train_set['Sales_date'].min(), train_set['Sales_date'].max()


# In[27]:


# checking completeness of dates 
difference = train_set['Sales_date'].max() - train_set['Sales_date'].min()
difference


# In[28]:


# Expected dates in dataset
difference.days + 1


# In[29]:


#Actual date in dataset 
train_set.Sales_date.nunique()


# In[30]:


# checking the range of dates 
#test_set.date.min(), test_set.date.max()


# In[31]:


# Adding the sales date column to the train dataset 
#test_set['Sales_date'] = pd.to_datetime(test_set['date']).dt.date


# In[32]:


# Checking for unique values in the test data set
#test_set.Sales_date.nunique()


# In[33]:


#checking the range of dates 
#test_set['Sales_date'].min(), test_set['Sales_date'].max()


# In[34]:


# checking completeness of dates 
#difference1 = test_set['Sales_date'].max() - test_set['Sales_date'].min()
#difference1


# In[35]:


# Expected dates in dataset
#difference1.days + 1


# In[36]:


#Actual date in dataset 
#test_set.Sales_date.nunique()


# ### We can see that there are missing dates in the actual number of dates available in the train set. Let's go on to show the missing dates. 
# 

# In[37]:


# Finding the head and tail of dates
expected_dates = pd.date_range(start= train_set['Sales_date'].min(), end = train_set['Sales_date'].max())
expected_dates


# In[38]:


# Finding the missing dates in both sets
set(expected_dates.date) - set(train_set.Sales_date.unique())


# In[39]:


train_set.head(5)


# In[40]:


# Importing products from itertools
from itertools import product


# In[41]:


missing_dates = set(expected_dates.date) - set(train_set.Sales_date.unique())
unique_stores = train_set.store_nbr.unique()
unique_families = train_set.family.unique()


# In[42]:


# Finding missing data
missing_data = list(product(missing_dates, unique_stores, unique_families))
missing_data


# In[43]:


new_data = pd.DataFrame(missing_data, columns = ['Sales_date', 'store_nbr', 'family'])
new_data


# # Merging Data

# In[44]:


# Merging trainset with new data which has filled missing data
merged_data = pd.concat([train_set, new_data], ignore_index=False)
merged_data


# In[45]:


# Dropping columns "id" and "date" in merged data column
merged_data.drop(columns = ["id", "date"], inplace = True)


# In[46]:


merged_data


# In[47]:


# Adding the City column to the merged data since I want to find the impact of promotions of sales in different cities 
merged_data['City'] = 'NaN'


# In[48]:


merged_data


# In[49]:


merged_data["City"] = merged_data['store_nbr'].map({54: 'El Carmen',53: 'Manta',52: 'Manta',51: 'Guayaquil', 50: 'Ambato',
                                              49:'Quito', 48: 'Quito',47: 'Quito', 46: 'Quito', 45: 'Quito', 45: 'Quito',
                                             44:'Quito', 43:'Esmeraldas', 42:'Cuenca', 41:'Machala', 40:'Machala', 39:'Cuenca',
                                             38:'Loja', 37:'Cuenca',36:'Libertad', 35: 'Playas', 34:'Guayaquil', 33:'Quevedo', 
                                              32:'Guayaquil', 31:'Babahoyo', 30:'Guayaquil', 29:'Guayaquil', 28:'Guayaquil',
                                              27:'Daule', 26:'Guayaquil',25:'Salinas', 24:'Guayaquil', 23:'Ambato', 22:'Puyo',
                                             21:'Santo Domingo', 20:'Quito', 19:'Guaranda', 18:'Quito', 17:'Quito', 16:'Santo Domingo',
                                             15:'Ibarra', 14:'Riobamba', 13:'Latacunga',12:'Latacunga', 11:'Cayambe', 10:'Quito',9:'Quito',
                                             8:'Quito',7:'Quito',6:'Quito',5:'Santo Domingo',4:'Quito', 3:'Quito', 2:'Quito', 1:'Quito'})
merged_data


# In[50]:


merged_data = merged_data.fillna(0)
merged_data


# In[51]:


sample_1 = merged_data[merged_data['onpromotion'] !=0].sort_values('onpromotion', ascending = False)
sample_1


# In[52]:


sample_1.head()


# In[53]:


stores.head()


# In[54]:


sample_1


# # DATA VISUALIZATION

# # Q1. Did promotions influence sales in some cities than the others?

# In[55]:


x = sample_1['City'].value_counts()[:10].index
y = sample_1['City'].value_counts()[:10].values
plt.bar(x,y)
plt.rcParams['figure.figsize'] = (25,10)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Promotion Influence', fontsize = 19, fontweight = 'bold')
plt.xlabel('Cities', fontsize = 19, fontweight = 'bold')
plt.ylabel('Sales', fontsize = 19, fontweight = 'bold')


# ### This graph represents the influence of promotion on sales in some specific cities. It is evident that sales in Quito is the highest when items are on promotion, followed by Guayaquil and Cuenca. 

# 

# In[56]:


sample_1


# In[57]:


sample_2 = merged_data[merged_data['onpromotion'] ==0].sort_values('onpromotion', ascending = False)
sample_2


# In[58]:


c = sample_2['City'].value_counts()[:10].index
d = sample_2['City'].value_counts()[:10].values
plt.bar(c,d)
plt.rcParams['figure.figsize'] = (25,10)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title(' NO PROMOTION ', fontsize = 19, fontweight = 'bold')
plt.xlabel('Cities', fontsize = 19, fontweight = 'bold')
plt.ylabel('Sales', fontsize = 19, fontweight = 'bold')


# ### Also, when there is no promotion, Quito still records the highest sales mainly because it has the most number of stores hence more patronage. To promote more sales in Ambato, Puyo and Guaranda, more promotions will have to be done.

# # Q2. Are sales affected by promotions and oil prices?
# 

# # OIL

# In[59]:


# Viewing oil dataset
oil.head()


# In[60]:


# Changing oil date to datetime 
oil['Sales_date'] = pd.to_datetime(oil['date']).dt.date


# In[61]:


# checking for the completeness of dates for oil prices.
date_difference = oil['Sales_date'].max() - oil['Sales_date'].min()
date_difference


# In[62]:


date_difference.days + 1


# In[63]:


# checking the unique number of days 
oil.Sales_date.nunique()


# In[64]:


#checking the range of dates 
oil['Sales_date'].min(), oil['Sales_date'].max()


# In[65]:


# checking for the expected days 
dates_expected = pd.date_range(start= oil['Sales_date'].min(), end = oil['Sales_date'].max())
dates_expected


# In[66]:


# Finding the missing dates in both sets
dates_missing = set(dates_expected.date) - set(oil.Sales_date.unique())
dates_missing


# In[67]:


missing_oil_data = list(product(dates_missing))


# In[68]:


# adding the missing oil data to get a new oil data with complete dates 
revised_oil_data = pd.DataFrame(missing_oil_data, columns = ['Sales_date'])
revised_oil_data


# In[69]:


# Merging oil and revised data
Merged_oil_data = pd.concat([oil, revised_oil_data], ignore_index=False)
Merged_oil_data


# In[70]:


# Filling in missing values in Merged_oil_data
Merged_oil_data = Merged_oil_data.ffill().bfill()
Merged_oil_data


# In[71]:


# Dropping the date column in Merged_oil_data
Merged_oil_data.drop(columns = ["date"], inplace = True)


# In[72]:


# Previewing merged oil data
Merged_oil_data


# In[73]:


# Concatenating Merged_oil_data and merged data to analyse the effect of oil prices on sales.
Merged_oil_data2 = merged_data.merge(Merged_oil_data, how='inner', on='Sales_date')
Merged_oil_data2


# In[74]:


# Checking for null values in merged data.
Merged_oil_data2.isnull().sum()


# In[75]:


Sp = Merged_oil_data2.plot.scatter(x="dcoilwtico", y="sales", c='green')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('dcoilwtico', fontsize = 25, fontweight = 'bold')
plt.ylabel('sales', fontsize = 25, fontweight = 'bold')


# 

# # Promotion

# In[76]:


# adding a column called PROMO/NOPROMO to the merged data set
merged_data['PROMO/NOPROMO']='NaN'
merged_data


# In[77]:


# mapping the merged data set to split the Promo/NoPromo column 
merged_data["PROMO/NOPROMO"] = merged_data['onpromotion'].map(
                           {0:'No Promo'})
merged_data


# In[78]:


# Replacing null values in the Promo/No Promo column with Promo
merged_data["PROMO/NOPROMO"].replace(np.NAN ,value='Promo', inplace=True)
merged_data


# In[79]:


sales_onpromo = merged_data.groupby(by = 'PROMO/NOPROMO').sales.agg(
    ["sum"]).sort_values(by = ["sum"], ascending = False)
sales_onpromo


# In[80]:


sales_onpromo.plot(kind ='bar')
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.ylabel("SALES",fontsize = 25,fontweight = 'bold')
plt.xlabel("PROMO and NO PROMO",fontsize = 20,fontweight = 'bold')
plt.legend(bbox_to_anchor =(1,1),fontsize = 15)
plt.title("SALES WITH AND WITHOUT PROMO",fontsize = 15,fontweight = 'bold')


# # Q3. Which items have the highest sales /lowest sales ?
# 

# In[81]:


train_set.family.unique()


# In[82]:


# Viewing the merged dataset 
merged_data


# In[83]:


# sorting the family column from highest to lowest
group_by_family = merged_data.groupby(by = "family").sales.agg(["sum"]).sort_values(by = ["sum"],ascending = False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
group_by_family


# In[84]:


plt.figure(figsize = (20,15))
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
sns.barplot(y = group_by_family[:10].index, x = (group_by_family["sum"])[:10])
plt.ylabel("Items sold",fontsize = 40,fontweight = 'bold')
plt.xlabel("Sales",fontsize = 25,fontweight = 'bold')
plt.title("ITEMS WITH HIGHEST AND LOWEST SALES",fontsize = 40,fontweight = 'bold')


# ### We can see that Grocery has the highest sales across all stores. In order to increase patronage for the items that have lower sales, Favorita can run more promotions on those items. We have seen the effect of promotions form the previous visualizations above. 

# #  Q4. Which date records the highest and lowest sales ? 

# In[85]:


# Previewing merged data set 
merged_data


# In[86]:


# Sorting sales values from highest to lowest
sample_3 = merged_data[merged_data['sales'] > 1].sort_values('sales', ascending = False)
sample_3


# In[87]:


merged_data['Sales_date'] = pd.to_datetime(merged_data['Sales_date']).dt.date


# In[88]:


# Finding the aggregated sales per date
aggregated_sales_perdate = sample_3.groupby('Sales_date', as_index=False)['sales'].sum()
aggregated_sales_perdate


# In[89]:


# Viewing aggregated sales data
aggregated_sales_perdate


# In[90]:


# Adding year, month, week and day columns to the Sample dataset
sample_3['year'] = pd.to_datetime(sample_3['Sales_date']).dt.year
sample_3['month'] = pd.to_datetime(sample_3['Sales_date']).dt.month
sample_3['week'] = pd.to_datetime(sample_3['Sales_date']).dt.week
sample_3['day'] = pd.to_datetime(sample_3['Sales_date']).dt.day
warnings.filterwarnings('ignore')


# In[91]:


# Viewing the sample_3 dataset
sample_3


# In[92]:


#Adding years column to the aggregated_sales_perdate column
aggregated_sales_perdate['years'] = pd.to_datetime(aggregated_sales_perdate['Sales_date']).dt.year
aggregated_sales_perdate.groupby(['years'], as_index=False)['sales'].max()
aggregated_sales_perdate


# In[93]:


# Filtering the maximum sales values
Maximum_values = aggregated_sales_perdate.groupby(['years'], as_index=False)['sales'].max()


# # For Highest Sales Date

# In[94]:


a = aggregated_sales_perdate.sort_values(by='sales', ascending=False)
b = a.loc[a.groupby("years")["sales"].idxmax()]


# In[95]:


for row in Maximum_values.itertuples():
    musk = (aggregated_sales_perdate['years'] == row.years) & (aggregated_sales_perdate['sales'] == row.sales)
    
    Maximum_values_row = aggregated_sales_perdate.loc[musk]
    
    #t = tmp_row['sales_date'].values
    #s = row.sales
    
    print('Peak sales for', row.years, 'occured on', Maximum_values_row['Sales_date'].values, 'for', row.sales, 'items')
    


# In[96]:


a


# In[97]:


b


# In[98]:


b.plot(x = 'Sales_date', y = 'sales', kind = 'bar')
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.ylabel("Sales made",fontsize = 30,fontweight = 'bold')
plt.xlabel("Sales date",fontsize = 30,fontweight = 'bold')
plt.title("HIGHEST SALES DATE",fontsize = 40,fontweight = 'bold')


# ### From the visualization above, the date that records the highest sales is 01/04/2017. 

# # For Lowest Sales Date 

# In[99]:


e = aggregated_sales_perdate[aggregated_sales_perdate['sales'] !=0].sort_values(by='sales', ascending=True)
e


# In[100]:


f = e.loc[e.groupby("years")["sales"].idxmin()]
f


# In[101]:


f.plot(x = 'Sales_date', y = 'sales', kind = 'bar')
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.ylabel("Sales made",fontsize = 30,fontweight = 'bold')
plt.xlabel("Sales date",fontsize = 30,fontweight = 'bold')
plt.title("LOWEST SALES DATE",fontsize = 40,fontweight = 'bold')


# # Q5. Did earthquake impact sales in any way?

# ### To answer this question, I am going to find the average of sales before and after the earthquake to determine the impact of the earthquae on sales. The earthquake occured on the 2016-04-16. 

# In[102]:


# Average sales before earthquake, i.e, 2016-04-01 to 2016-04-15
merged_data['Sales_date'] = pd.to_datetime(merged_data['Sales_date']).dt.date


# In[103]:


merged_data.info()


# In[104]:


merged_data['Sales_date']= merged_data['Sales_date'].astype('str')


# In[105]:


before_earthquake = merged_data[(merged_data['Sales_date']> '2016-04-01') & (merged_data['Sales_date']<= '2016-04-15')]


# In[106]:


#finding the average of sales for before earthquake.
avg_before_earthquake = before_earthquake['sales'].mean()
avg_before_earthquake


# In[107]:


# Average sales after earthquake, i.e, 2016-04-17 to 2016-04-30
after_earthquake = merged_data[(merged_data['Sales_date']> '2016-04-17') & (merged_data['Sales_date']<= '2016-04-30')]
#finding the average of sales for before earthquake.
avg_after_earthquake = after_earthquake['sales'].mean()
avg_after_earthquake


# ### We have the average of sales before the earthquake to be 445 and that of after te earthquake to be 511.7. This explains that, there was more sales after the earthquake than before the earthquake. We can conclude from the averages that earthquake in a way had a positive impact on sales. 

# # Q6.Which holidays affect sales the most?

# In[108]:


# Viewing the holiday dataset
holidays.head()


# In[109]:


# Printing a concise summary of the holidays data set
holidays.info()


# In[110]:


holidays['Sales_date'] = pd.to_datetime(holidays['date']).dt.date


# In[111]:


holidays


# In[112]:


merged_data


# In[113]:


holidays.info()


# In[114]:


merged_data.info()


# In[115]:


# Changing both the merged data and holidays data to datetime
merged_data['Sales_date']= merged_data['Sales_date'].astype('str')
holidays['Sales_date']= holidays['Sales_date'].astype('str')


# In[116]:


# Merging the holidays dataset with the Merged data
Merged_holiday_set = merged_data.merge(holidays, how='inner', on='Sales_date')
Merged_holiday_set


# In[117]:


# Dropping the date column
Merged_holiday_set.drop(columns = ["date"], inplace = True)
Merged_holiday_set


# In[118]:


sns.countplot(Merged_holiday_set['locale'])
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Holiday', fontsize = 22, fontweight = 'bold')
plt.ylabel('Sales', fontsize = 22, fontweight = 'bold')
plt.title('Holidays and Sales', fontsize = 22, fontweight = 'bold')
plt.rcParams['figure.figsize'] = (18,8)
warnings.filterwarnings('ignore')


# # Feature Engineering 

# # Train

# In[119]:


# Merging the train_set with the stores data set
Train = pd.merge(train_set, stores)
Train


# In[120]:


#MErging the new Train set with oil data
Train = pd.merge(Train,Merged_oil_data, on = 'Sales_date')
Train.info()


# In[121]:


# checking for completeness of date for holidays data
expected_dates = pd.date_range(start= train_set['Sales_date'].min(), end = train_set['Sales_date'].max())
expected_dates


# In[122]:


# Finding the missing holidays dates
missing_holiday_dates = set(expected_dates.date) - set(holidays["date"].unique())
missing_holiday_dates


# In[123]:


# adding the missing holidays date
holidays_add = pd.DataFrame(missing_holiday_dates, columns = ["date"])
holidays_add


# In[124]:


# renaming columns
holidays_add.rename(columns = {"date":"Sales_date"}, inplace = True)
holidays_add


# In[125]:


#dropping columns
holidays.drop(columns = ["date",'description'], inplace = True)
holidays.head()


# In[126]:


holidays.rename(columns = {"Sales_date":"sales_date"}, inplace = True)
holidays


# In[127]:


# changing holidays date to datetime
holidays['Sales_date'] = pd.to_datetime(holidays['sales_date']).dt.date


# In[128]:


# merging data
holidays = holidays_add.merge(holidays ,how='left', on='Sales_date')
#holidays.head()


# In[129]:


#holidays.isnull().sum()


# In[130]:


# merging the Train data with complete holidays dataset
Train = pd.merge(Train, holidays, on = "Sales_date")
Train


# In[131]:


# Checking for nll values
Train.isnull().sum()


# In[132]:


# Filling the nulls in the holiday data
Train["type_y"] = Train["type_y"].fillna("Work Day")
Train["locale"] = Train["locale"].fillna("National")
Train["transferred"] = Train["transferred"].fillna(False)
Train.isnull().sum()


# In[133]:


Train


# In[134]:


# chamging the dates column in transactions dataset to datetime. 
transactions['Sales_date'] = pd.to_datetime(transactions['date']).dt.date


# In[135]:


# merging Train data with transactions data
Train = pd.merge(Train, transactions, on = ["Sales_date", "store_nbr"])
Train.info()


# In[136]:


Train


# In[137]:


# Dropping columns
Train.drop(columns = ['date_x', 'type_x', 'type_y', 'date_y', 'sales_date','locale_name', 'id'], inplace = True)


# In[138]:


Train.to_csv("data.csv")


# In[139]:


Train.info()


# In[ ]:





# In[140]:


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# In[141]:


# Scaling 
cols_to_scale = ['dcoilwtico']
scaler = MinMaxScaler()
Train[cols_to_scale] = scaler.fit_transform(Train[cols_to_scale])


# In[142]:


# Encoding the categorical columns
categoricals = Train.select_dtypes(include=["object"]).columns.to_list()
categoricals.remove("Sales_date")
categoricals


# In[143]:


# Encoding the categorical variables
oh_encoder = OneHotEncoder(drop = "first", sparse = False)
oh_encoder.fit(Train[categoricals])
encoded_categoricals = oh_encoder.transform(Train[categoricals])
encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = oh_encoder.get_feature_names_out().tolist())
encoded_categoricals


# In[144]:


# Adding the encoded categoricals to the DataFrame and dropping the original columns
final_train = Train.join(encoded_categoricals)
final_train.drop(columns= categoricals, inplace= True)
final_train


# In[145]:


def getSeason(row):
    if row in (3,4,5):
        return 'Spring'
    elif row in (6,7,8):
        return 'Summer'
    elif row in (9,10,11):
        return 'Fall'
    elif row in (12,1,2):
        return 'Winter'
    
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df['season'] = df['month'].apply(getSeason)
    
    return df


# In[146]:


# Getting date features for train data set
final_train = getDateFeatures(final_train, 'Sales_date')


# In[147]:


pd.set_option('display.max_columns', None)


# In[148]:


final_train.head()


# In[149]:


final_train.drop(columns = ['date','transferred','season'],inplace= True)


# In[150]:


final_train


# In[151]:


final_train.drop(columns = ['Sales_date'], inplace = True)


# In[152]:


final_train.columns.to_list()


# In[153]:


# finding the correlation
final_train.corr()


# In[154]:


plt.figure(figsize = (14,14))
sns.heatmap(final_train.corr(), vmin = -1 , cmap = 'YlGnBu')


# In[155]:


#final_train.drop(columns = ['season'], inplace = True)


# In[156]:


final_train.drop(columns = ['transactions'], inplace = True)
final_train


# # MODELING 

# In[157]:


final_train[final_train["year"] <= 2016]


# In[158]:


# Splitting the data into train and test with years 
train = final_train.loc[(final_train['year'].isin([2013, 2014, 2015, 2016]) & final_train['year'].isin([2013, 2014, 2015, 2016]))]
test = final_train.loc[(final_train['year'].isin([2017]) & final_train['year'].isin([2017]))]


# In[159]:


#test.year.unique(), test.month.unique()
test.year.unique()


# In[160]:


train[train['year'] <= 2016].year.unique()


# # Using the DecisionTreeRegressor Model

# In[161]:


decision_tree_model = DecisionTreeRegressor(random_state = 24)


# In[162]:


X_train = train.drop(columns = ['sales'])
Y_train = train['sales']


# In[224]:


X_train.columns.to_list()


# In[163]:


decision_tree_model.fit(X_train,Y_train)


# In[164]:


X_test = test.drop(columns = ['sales'])
Y_test = test['sales']


# In[165]:


y_pred = decision_tree_model.predict(X_test)


# In[166]:


mean_absolute_error(Y_test,y_pred)


# In[167]:


np.sqrt(mean_squared_error(Y_test,y_pred))


# In[168]:


y_pred


# In[169]:


np.sqrt(mean_squared_log_error(Y_test, y_pred))


# In[170]:


## get importance
dt_importance = decision_tree_model.feature_importances_
dt_importance = pd.DataFrame(dt_importance, columns = ["score"]).reset_index()
dt_importance["Feature"] = list(X_train.columns)
dt_importance.drop(columns = ["index"], inplace = True)
dt_importance.sort_values(by = "score", ascending = False).head()


# # Using the Linear Regression Model

# In[171]:


linear_model = LinearRegression()


# In[172]:


linear_model.fit(X_train,Y_train)


# In[173]:


y_pred = linear_model.predict(X_test)


# In[174]:


mean_absolute_error(Y_test,y_pred)


# In[175]:


np.sqrt(mean_squared_error(Y_test,y_pred))


# In[176]:


#np.sqrt(mean_squared_log_error(Y_test, y_pred))


# In[177]:


importance = linear_model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[178]:


# getting feature importance
importance = pd.DataFrame(importance, columns = ["score"]).reset_index()
importance["Feature"] = list(X_train.columns)
importance.drop(columns = ["index"], inplace = True)
importance.sort_values(by = "score", ascending = False)
importance


# In[179]:


fig = px.bar(importance, x = "Feature", y = "score")
fig.show()


# In[180]:


# Evaluate Linear model RMSE
scores = cross_val_score(linear_model, X_train, Y_train, scoring = "neg_mean_squared_error", cv = 5, n_jobs = 1)
rmse = np.sqrt(-scores)
print("RMSE values: ", np.round(rmse, 2))
print("RMSE average: ", np.mean(rmse))


# In[181]:


#np.sqrt(mean_squared_log_error(Y_test, y_pred))


# # Using the RandomForestRegressor

# In[182]:


rf = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, Y_train)


# In[183]:


rf


# In[184]:


y_pred = rf.predict(X_test)


# In[185]:


mean_absolute_error(Y_test,y_pred)


# In[186]:


np.sqrt(mean_squared_error(Y_test,y_pred))


# In[187]:


np.sqrt(mean_squared_log_error(Y_test, y_pred))


# # TEST

# In[188]:


# Viewing the test dataset
test_set


# In[189]:


# Checking if there are any missing dates
test_set_range = test_set.date.min(), test_set.date.max()
test_set_range


# In[190]:


# Number of expected dates
expected_test_days = pd.date_range(start = test_set["date"].min(), end = test_set["date"].max())
expected_test_days


# In[191]:


#Previewing holiday data 
holidays


# In[192]:


# Getting missing dates
holidays.rename(columns = {"sales_date":"date"}, inplace = True)


# In[193]:


# Finding missing holiday dates
missing_holiday_dates = set(expected_test_days.date) - set(holidays["date"].unique())
missing_holiday_dates


# In[194]:


# Creating a dataframe for the missing dates in the holiday data
holidays_addition = pd.DataFrame(missing_holiday_dates, columns = ["date"])
holidays_addition


# In[195]:


# Adding the  missing holiday dates to the main dataframe
holidays = pd.concat([holidays, holidays_addition], ignore_index=True)
holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
holidays


# In[196]:


# Filling in missing values with variables  
holidays["type"] = holidays["type"].fillna("Work Day")
holidays["locale"] = holidays["locale"].fillna("National")
holidays["locale_name"] = holidays["locale_name"].fillna("Ecuador")
holidays["transferred"] = holidays["transferred"].fillna(False)


# In[197]:


# Merging test set with stores 
Test = pd.merge(test_set, stores)
Test


# In[198]:


# Previewing oil dataset
oil


# In[199]:


# Getting missing dates
missing_oil_dates = set(expected_test_days.date) - set(oil["date"].unique())
missing_oil_dates


# In[200]:


# Adding the  missing oil dates to the main dataframe
oil_dates_add = pd.DataFrame(missing_oil_dates, columns = ["date"])
oil = pd.concat([oil, oil_dates_add], ignore_index=True)
oil["Sales_date"] = pd.to_datetime(oil["date"])
oil = oil.sort_values(by = ["Sales_date"], ignore_index = True)
oil.head()


# In[201]:


# Filling nulls with forward fill and backfill
oil = oil.ffill().bfill()
oil


# In[202]:


Test


# In[203]:


Test["Sales_date"] = pd.to_datetime(Test["date"])


# In[204]:


# Merging the train data with the other dataframes
Test = pd.merge(Test, oil, on = 'Sales_date')
Test


# In[205]:


holidays['Sales_date'] = pd.to_datetime(holidays["date"])


# In[206]:


# Merging the Test and holidays dfs
Test = pd.merge(Test, holidays, on = "Sales_date")
Test


# In[207]:


# Dropping columns 
Test.drop(columns = ['id', 'date_x','type_x', 'date_y', 'date','type_y'], inplace = True)
Test


# In[208]:


Test


# In[209]:


Test.drop(columns = ['transferred','locale_name'], inplace = True)
Test


# In[210]:


encoded_categoricals = oh_encoder.transform(Test[categoricals])
encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = oh_encoder.get_feature_names_out().tolist())
encoded_categoricals


# In[211]:


# Adding the encoded categoricals to the DataFrame and dropping the original columns
final_test = Test.join(encoded_categoricals)
final_test.drop(columns= categoricals, inplace= True)
final_test


# In[212]:


# Defining a function to get date features from dataframe
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df['season'] = df['month'].apply(getSeason)
    
    return df


# In[213]:


# Getting date features matched with Sales
final_test = getDateFeatures(final_test, 'Sales_date')


# In[214]:


pd.set_option('display.max_columns', None)


# In[215]:


final_test.drop(columns = ['Sales_date'], inplace = True)
final_test


# In[216]:


final_test.drop(columns = ['date'], inplace = True)
final_test


# In[217]:


final_test


# In[218]:


#X_predict = final_test[["store_nbr", "onpromotion", "dcoilwtico", "day_of_week",
                        #"day_of_month","is_month_start","is_month_end","day_of_year", "is_weekend", 
                  #"week_of_year", "month", "year","quarter","city_Babahoyo","city_Cayambe", "city_Cuenca","city_Daule", "city_Esmeraldas" ,"city_Guayaquil", 
                  #"city_Libertad", "city_Machala","city_Playas", "city_Quevedo", "city_Quito", "city_Salinas","city_Santo Domingo"]]
#X_predict = final_test[['store_nbr'	'onpromotion'	'cluster'	'dcoilwtico'	'locale_name'	'month'	'day_of_month', 
                        #'day_of_year'	'week_of_year'	'day_of_week'	'year'	'is_weekend',
                       #'is_month_start'	'is_month_end'	'quarter'	'is_quarter_start'	'is_quarter_end'	,
                       # 'is_year_start'	'is_year_end'	'season'	'family_BABY' 'CARE'	'family_BEAUTY'	,
                       #'family_BEVERAGES'	'family_BOOKS'	'family_BREAD'/BAKERY	family_CELEBRATION	family_CLEANING	,
                       # family_DAIRY	family_DELI	family_EGGS	family_FROZEN FOODS,
                      # family_GROCERY I	family_GROCERY II	family_HARDWARE	family_HOME AND KITCHEN I	family_HOME AND KITCHEN II	,
                       # family_HOME APPLIANCES	family_HOME CARE	family_LADIESWEAR	family_LAWN AND GARDEN	family_LINGERIE,
                       #family_LIQUOR,WINE,BEER	family_MAGAZINES	family_MEATS	family_PERSONAL CARE	family_PET SUPPLIES	,
                      #  family_PLAYERS AND ELECTRONICS	family_POULTRY	family_PREPARED FOODS	family_PRODUCE, 
                      #amily_SCHOOL AND OFFICE SUPPLIES	family_SEAFOOD	city_Babahoyo	city_Cayambe	city_Cuenca	city_Daule,	
                      #  city_El Carmen	city_Esmeraldas	city_Guaranda	city_Guayaquil,	
                      #  city_Ibarra	city_Latacunga	city_Libertad	city_Loja	city_Machala	city_Manta	city_Playas	city_Puyo,	
                      #  city_Quevedo	city_Quito	city_Riobamba	city_Salinas,	
                       # city_Santo Domingo	state_Bolivar,	
                       # state_Chimborazo	state_Cotopaxi	state_El Oro	state_Esmeraldas	state_Guayas	state_Imbabura,	
                       # state_Loja	state_Los Rios	state_Manabi	state_Pastaza,	
                       # state_Pichincha	state_Santa Elena,	
                       # state_Santo Domingo de los Tsachilas	state_Tungurahua	locale_National	locale_Regional]]







# In[219]:


#fitting the chosen model to the training data and making predictions 
decision_tree_model = DecisionTreeRegressor(random_state = 24)
decision_tree_model = decision_tree_model.fit(X_train,Y_train)
y_pred = decision_tree_model.predict(X_test)
#y_pred = pd.DataFrame(decision_tree_model.predict(X_test))


# # Exportation 
# 

# In[220]:


requirements = '\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None))

with open('requirements.txt', 'w') as f:
    f.write(requirements)


# In[221]:


to_export = {
    "encoder": oh_encoder,
    "scaler": scaler,
    "model": decision_tree_model,
    "pipeline": None,
}


# In[222]:


import os,pickle


# In[223]:


with open('ML_items', 'wb') as file:
    pickle.dump(to_export, file)


# In[ ]:




