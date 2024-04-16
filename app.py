"""
Université Virtuelle du Tchad.
Master Securité Logicielle.
LAGRE GABBA BERTRAND (https://github.com/FoubaDev/)
"""
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu



@st.cache_resource
def load_data_csv():
   df = pd.read_csv("Breast_cancer_data.csv")
   return df

@st.cache_resource
def load_data_sav():
   loaded_model = joblib.load(open('Breast_cancer_data.sav', 'rb'))
   return loaded_model


df = load_data_csv()
loaded_model = load_data_sav()


@st.cache_resource
def predict(input_data):
    
    my_array = np.array(input_data)
    input_reshaped = my_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_reshaped)
    
    st.write("The prediction is :")
    st.write(prediction)
    
    

# create column for dashbaord
@st.cache_resource(experimental_allow_widgets=True)
def dashboard():

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown('https://github.com/FoubaDev')


def main():

    with st.sidebar:
        selected = option_menu ("Income Prediction System",
                            
                            ['Home',
                             'Prediction',
                            'Dataset',
                            'Authors',
                            ],
                            icons = ['house','bi bi-file-medical-fill','person','book','person'],
                            default_index=0
    
                            )
    
    if(selected == "Home") :
        st.title("Medical Diagnostic Prediction System")
        st.title("Let's defeat breast cancer together")
     
        dashboard()

    if(selected == "Dataset") :
        st.write(df)
        st.write(df.shape)
    if(selected == "Prediction") :
    
       
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
            mean_radius = st.number_input('mean_radius', min_value=0.0,) 
        
        with col2:
            mean_texture = st.number_input('mean_texture', min_value=0.0,) 
        
        with col3:
            mean_perimeter = st.number_input('mean_perimeter', min_value=0.0,) 
        
        with col4:
            mean_area = st.number_input('mean_area', min_value=0.0,) 
        
        with col5:
            mean_smoothness = st.number_input('mean_smoothness', min_value=0.0,) 


        BreathCancer = ''
        data = (mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, )   
        
        if st.button('Prediction'):
           
            result = predict(data)
        st.success(BreathCancer)
        
    if(selected == "Authors") :
         
         
        st.subheader("Authors :\n 1. ADOUM AHMAT GRENE \n\n2. BOKHIT ABDOULAYE DIGUI \n\n3. LAGRE GABBA BERTRAND \n\n4. MADJINGUESSOUM BRICE  " )
   
        st.write(" Software Security Students at  Chad Virtual University. ") 
        st.write("Github link  : https://github.com/FoubaDev/TpML_UVT.git \n")

        st.write(" It is our pleasure to see you reading our work.")
        st.write(" This practical work aims to solve a problem concerning the classification of patients with breast cancer and those without. Such a model will help hospitals in terms of decision making on new patients with the same clinical data.")
        
if __name__=='__main__':
    main()
