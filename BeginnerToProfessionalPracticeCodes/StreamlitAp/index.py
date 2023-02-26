import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("diabetes.csv")

st.title("Diabetes Prediction App")
#Headings
st.sidebar.header("Patient Data")
st.subheader("description stats of data")
st.write(df.describe())

age_option = df['Age'].unique().tolist()
Age = st.selectbox("which age should we plot to see corresponding glucose?", age_option, 0)
df = df[df['Age']==Age]

#plotting
fig = px.scatter(df, x='Glucose', y = 'Pregnancies', size='BloodPressure', color='BMI', hover_name = 'BMI',log_x=True, size_max=55, range_x=[50,200], range_y=[0,15]) 
st.write(fig)

#split data into x and y
X =df.drop(['Outcome'], axis=1)
y = df.iloc[ : , -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#function
def user_report():
    Pregnancies = st.sidebar.slider("Pregnancies", 0,17,2)
    glucose = st.sidebar.slider("glucose", 0,199,110)
    bp = st.sidebar.slider("blood_pressure", 0,122,72)
    sk = st.sidebar.slider("skin_thickness", 0,99,23)
    insulin = st.sidebar.slider("insulin", 0.0,846.0,30.5)
    bmi = st.sidebar.slider("bmi", 0.0,67.1,32.0)
    dpf = st.sidebar.slider("diabetes_pedigree_function", 0.078,2.42,0.3725)
    Age = st.sidebar.slider("Age", 21,81,29)

    user_report_data = {
        'Pregnancies': Pregnancies,
        'glucose': glucose,
        'bp': bp,
        'sk': sk,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'Age': Age
        }

    report_data=pd.DataFrame(user_report_data, index=[0])
    return report_data

#Patient data
user_data = user_report()
st.subheader("Patient Data")
st.write("user_data")

#model
rc = RandomForestClassifier()
rc.fit(X_train, y_train)
user_result = rc.predict(user_data)

#visualization
st.title("visualized patient data")

#color function
if user_result[0] == 0:
    color = "blue"
else:
    color = "red"

#Age vs pregnancies plot
st.header("Pregnancy count graph (other vs Yours")
fig_preg=plt.figure()
ax1 = sns.scatterplot(x = 'Age', y='Pregnancies', data= df, hue='Outcome', palette="Greens")
ax2 = sns.scatterplot(x = user_data["Age"], y=user_data['Pregnancies'], s= 150, color= color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0 -  Healthy & 1 - Diabetic")
st.pyplot(fig_preg)

#output
st.header("your Report: ")
output = ''
if user_result[0] == 0:
    output = 'You are Healthy'
    st.balloons()
else:
    output = 'You are Diabetic'
    st.warning("sugar,sugar,sugar")
    st.title(output)
#    st.subheader("accuracy: ")
 #   st.write(str(accuracy_score(y_test, rc.predict(X_test))*100 + "%"))
