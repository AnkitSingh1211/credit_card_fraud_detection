import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pickle
# Title of the app
st.title("Credit Card Transactions Anomaly Detection")



# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

iso=pickle.load(open('iso.pkl','rb'))

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    df=data.copy()

    # Sample DataFrame
    def object_detect(df):
        # Convert to numeric, coercing errors to NaN
        df = pd.to_numeric(df, errors='coerce')
        return df

    data['V3'] = data['V3'].apply(object_detect)
    data['V3'].fillna(data['V3'].mode()[0], inplace=True)

    data['V8'] = data['V8'].apply(object_detect)
    data['V8'].fillna(data['V8'].mode()[0], inplace=True)

    data['V16'] = data['V16'].apply(object_detect)
    data['V16'].fillna(data['V16'].mode()[0], inplace=True)


    # Display the data
    st.write("Data Preview:")
    st.write(data.head())



    # Predict anomalies
    data['anomaly'] = iso.predict(data)


    # Display the results
    st.write("Anomaly Detection Results:")
    st.write(data[['Time', 'Amount', 'anomaly']])

    # Visualize the anomalies
    st.write('Anomaly Visualization')
    fig, ax = plt.subplots()
    plt.scatter(data['Time'], data['Amount'], c=data['anomaly'], cmap='coolwarm')

    ax.set_xlabel('Time')
    ax.set_ylabel('Amount')

    st.pyplot(fig)
