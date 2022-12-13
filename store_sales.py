#importing libraries 
import streamlit as st 
import pandas as pd 
import numpy as np 
import os, pickle
from PIL import Image


# Setting containers
header = st.container()
dataset = st.container()
form = st.form(key="Input", clear_on_submit=True)

# Defining the path for importing Ml Items
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):

    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)

    return loaded_object

def getSeason(row):
    if row in (3,4,5):
        return 'Spring'
    elif row in (6,7,8):
        return 'Summer'
    elif row in (9,10,11):
        return 'Fall'
    elif row in (12,1,2):
        return 'Winter'

#Getting Date features    
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

# Loading the toolkit
loaded_toolkit = load_ml_toolkit("/Users/Admin/Desktop/post-bap-P4-master/ML_items")

if "results" not in st.session_state:

    st.session_state["results"] = []


# Instantiating the elements of the Machine Learning Toolkit
scaler = loaded_toolkit["scaler"]
decision_tree_model = loaded_toolkit["model"]
oh_encoder = loaded_toolkit["encoder"]     
# Setting header
with header:
    

    st.title('Store Sales forecasting - Corporation Favorita')
    #st.text('In this project I predicted store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer and to build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores')
    st.sidebar.write(
    f'In this project I predicted store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer and to build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores'
)

# Writing a function to load data
@st.cache()
def load_data(relative_path):
    train_data= pd.read_csv(relative_path, index_col= 0)
    train_data["Sales_date"] = pd.to_datetime(train_data["Sales_date"]).dt.date
    #train_data["year"]= pd.to_datetime(train_data['Sales_date']).dt.year
    
    return train_data

# Loading the base dataframe
rpath = "/Users/Admin/Desktop/Streamlit app folder/data.csv"
train_data = load_data(rpath)

with dataset:
    dataset.markdown("**This is the dataset of Corporation Favorita**")
    if dataset.checkbox("Preview the dataset"):
        dataset.write(train_data.head())

# Setting features andoutput columns
with form:
    left_column, right_column = st.columns(2)

    left_column.subheader('User Inputs')
    left_column.write('This section receives your input to be used for prediction')



# Designing the input section of my app
with form:
    Sales_date = left_column.date_input("Select a date:", min_value= train_data["Sales_date"].min())
    city = left_column.selectbox('City:', options = set(train_data['city']) )
    family = left_column.selectbox('Family:', options = set(train_data['family']))
    state = left_column.selectbox('State:',options = set(train_data['state']))
    store_nbr = left_column.selectbox('Store Number:', options = set(train_data['store_nbr']))
    cluster = right_column.selectbox('Store Cluster:',options = set(train_data['cluster']))
    onpromotion = right_column.number_input('No. of items on promotion:',min_value=train_data['onpromotion'].min(),max_value=train_data['onpromotion'].max(), value=train_data['onpromotion'].min())
    oil_price = right_column.number_input('Oil Price:',min_value=train_data['dcoilwtico'].min(),max_value=train_data['dcoilwtico'].max(), value=train_data['dcoilwtico'].min())
    locale = right_column.selectbox('Locale:',options = set(train_data['locale']))


 #Submit button
    submitted = st.form_submit_button(label="Submit")

# Logic when the inputs are submitted
if submitted:

    # Inputs formatting
    dict_input = {'Sales_date': [Sales_date],'city': [city],'family': [family],'state': [state], 
    'store_nbr': [store_nbr], 'cluster': [cluster], 'onpromotion': [onpromotion], 'dcoilwtico': [oil_price], 
    'locale':[locale]}

# Converting Input into a dataframe
    Input_data = pd.DataFrame.from_dict(dict_input)
    
# getting date features 
    Input_data = getDateFeatures(Input_data, 'Sales_date')

#Preprocessing 

# Scaling
    cols_to_scale = ['dcoilwtico']
    Input_data[cols_to_scale] = scaler.fit_transform(Input_data[cols_to_scale])

# Encoding 
    categoricals = ['family', 'city', 'state', 'locale']
    Input_data.drop(columns= ['Sales_date','date','season'], inplace = True)
    encoded_categoricals = oh_encoder.transform(Input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = oh_encoder.get_feature_names_out().tolist())
    Final_data = Input_data.join(encoded_categoricals)
    Final_data.drop(columns= categoricals, inplace= True)
    #st.write(Final_data.head())
    #st.write(Final_data.columns.to_list())
        
# Modeling
    output = decision_tree_model.predict(Final_data)

    Final_data["sales"] = output
    display = output[0]


    # Displaying prediction results
    st.success(f"**Predicted sales**: USD {display}")

# Adding the predictions to previous predictions
    st.session_state["results"].append(Final_data)
    result = pd.concat(st.session_state["results"],)

# Expander to display previous predictions
    st.expander("**Review previous predictions**")
    st.dataframe(result, use_container_width=True)
