import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
def upload_data():
    url =  'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    column_names = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species')
    data = pd.read_csv(url, header=None, names=column_names)
    return data

data = upload_data()

st.title('Dashboard for the iris data set')
st.write('this is a display and analysation of the iris data set')

if st.checkbox('show raw data'):
    st.subheader('raw data')
    st.dataframe(data)

#show the average sepal length for each species
st.subheader('Average sepal length for each species')
average_sepal_length = data.groupby('species')['sepal_length'].mean().reset_index()
fig = px.bar(average_sepal_length, x='species', y='sepal_length', color='species')
st.plotly_chart(fig)

# display a scatter plot displaying two features
st.subheader('compare teo features using a scatter plot')
st.write('choose two features in the iris data set to compare in a scatter plot')
feature1 = st.selectbox('choose the first feature', data.columns)
feature2 = st.selectbox('choose the second feature', data.columns)

figure2 = px.scatter(data, x=feature1, y=feature2, color='species', hover_name='species')
st.plotly_chart(figure2)

# filtering data according to species
st.subheader('filter data based on species')
st.write('choose species')
selected_species = st.multiselect('choose species', data['species'].unique())

if selected_species:
    filtered_species = data[data['species'].isin(selected_species)]
else:
    st.write('no species selected')

# display a pairplot for the selected species
if st.checkbox("Show pairplot for the selected species"):
    st.subheader("Pairplot for the Selected Species")

    if selected_species:
        fig2 = sns.pairplot(filtered_species, hue="species")
    else:
        sns.pairplot(data, hue="species")
        
    st.pyplot(fig2)

# show the distribution of a selected feature
st.subheader('distribution of selected feature')
selection = st.selectbox('choose a feature  for distribution', data.columns)
fig3 = px.histogram(data, x=selection, color='species', nbins=20)
st.plotly_chart(fig3)
