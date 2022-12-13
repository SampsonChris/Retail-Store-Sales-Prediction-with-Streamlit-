---

# Sales Prediction App with Streamlit

Sales forecasting is one of the most important things a company does. It fuels sales planning and is used throughout an enterprise for staffing and budgeting. Despite its importance, many organisations use outmoded practices that produce bad forecasts.
Producing an accurate sales forecast is vital to business success. Public companies can quickly lose credibility if they miss a forecast. Machine learning is said to be one of the most accurate ways to forecast sales. In this article, I am going to demonstrate how I used Streamlit to deploy my already built machine learning model. A link to the source code of the built model will be available in this article. 

---

## Introduction to Streamlit
Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.
Streamlit turns data scripts into shareable web apps in minutes.
All in pure Python. No front‑end experience required. 
To install streamlit, kindly follow the steps below; 

### Installing
pip install streamlit

### Testing if installation worked
streamlit hello

### In your virtual environment
streamlit run myfile.py # <---- you can change "myfile.py" to suit your file path.

## Processes taken in building my Streamlit app

1.Exporting Machine Learning(ML) items
2.Setting  up environment
3.Importing  Machine Learning items from local PC
4.Building the app Interface
5.Setting up the backend to process inputs and display outputs
6.Deployment

## 1. Exporting Machine Learning(ML) items
This is the first step I took in processing my Streamlit app. Exports are taken from my initial Jupyter notebook. A link will be provided as stated earlier to view the source codes. The ML items exported include ; the chosen model, encoder, scaler if used in the notebook and also a pipeline if available. These various items can be exported individually but for ease of access, I created a dictionary to export the ML Items at a go. Pickle was used in exporting the ML items. Also, OS is going to be a useful tool in exporting the requirements as well. 
Below is an illustration of how to export the ML items using dictionary and pickle. 
Import os, pickle

#Exporting the requirements
requirements = "\n".join(f"{m.__name__}=={m.__version__}" for m in globals().values() if getattr(m, "__version__", None))

with open("requirements.txt", "w") as f:
    f.write(requirements)

#creating a dictionary of exports
to_export = {
    "encoder": oh_encoder,
    "scaler": scaler,
    "model": decision_tree_model,
    "pipeline": None,
}

#exporting ML items
with open('ML_items', 'wb') as file:
    pickle.dump(to_export, file)

## 2. Setting up environment
This step involved creating a repository or folder for exported items. A python script was also be needed to host the backend codes for the success of the app. I created a virtual environment to prevent any disputes with any related variables. Below is the code I used in activating my virtual environment and setting up up my streamlit
#Creating and activating virtual environment
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

To know if your virtual environment is successfully activated, you will see (venv) prior to the current path in the active terminal. 

## 3. Importing Machine Learning items from local PC
After the virtual environment is set, we then Import the machine learning toolkit. Below is the code I used in loading my toolkit into active python script. 
 #Defining the path for importing Ml Items
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):

    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)

    return loaded_object
    
The use of the @st.cache function decorator is to memoize function executions. It simply means it overrides previous ran codes and makes loading of data easier and faster. 
After importing the ML items from local PC, you then instantiate the elements of the toolkit. Mind you, they are the elements which were exported earlier from the Jupyter notebook. Below is the code I used in Instantiating my ML tool kit elements 
#Instantiating the elements of the Machine Learning Toolkit
scaler = loaded_toolkit["scaler"]
decision_tree_model = loaded_toolkit["model"]
oh_encoder = loaded_toolkit["encoder"]

## 4. Building the app Interface
This step is where I employed the use of Streamlit's various components to build my Interface. The most common components I used as stated below;
st.container(): to define a container (read box) to keep other components and keep your work organised
st.form(): allows you to create a form to receive inputs from users
st.columns(n): to define columns in your workspace. "n" could be replaced with the number of columns you want to create
st.date_input(): to receive date inputs
st.selectbox(): for a dropdown box
st.sidebar: for a sidebar
st.number_input(): for number inputs
st.radio(): for a radio
st.checkbox(): for a checkbox


## 5. Setting up the backend to process inputs and display outputs
This section replicates the steps taken in my initial Jupyter notebook. After building your interface, you then build the backend to make prediction. As stated in the early lines of this section, the steps taken in the Jupyter notebook are replicated, ie, 
a. Receiving inputs 
b. Encoding categorical columns 
c. Scaling numerical columns 
d. Predicting and returning the output of predictions. 
The processes taken in this step are shown in the link provided in my Github repository. 

## 6. Deployment 
To deploy the built app, visit https://streamlit.io/cloud, sign in and connect your GitHub . You can then select new app and the repository of the app for deployment.

I hope this article was helpful. ☺️

This is a link to my github repo 

## References
Streamlit Docs
Streamlit is more than just a way to make data apps, it's also a community of creators that share their apps and ideas…docs.streamlit.io
