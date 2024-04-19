"""
Université Virtuelle du Tchad.
Master Cybersécurité.
(https://github.com/FoubaDev/)
"""
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu



@st.cache_resource(experimental_allow_widgets=True)
def load_data_csv():
   df = pd.read_csv("Breast_cancer_data.csv")
   return df

@st.cache_resource(experimental_allow_widgets=True)
def load_data_sav():
   loaded_model = joblib.load(open('Breast_cancer_data.sav', 'rb'))
   return loaded_model


df = load_data_csv()
loaded_model = load_data_sav()


@st.cache_resource(experimental_allow_widgets=True)
def predict(input_data):
    
    my_array = np.array(input_data)
    input_reshaped = my_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_reshaped)
        
    st.write("The prediction is :")
    st.write(prediction)
    
@st.cache_resource(experimental_allow_widgets=True)
def visualization():
    st.title('Vue descriptive du dataset ')
    st.write(df.describe())
    
        
        
   

# create column for dashbaord
@st.cache_resource(experimental_allow_widgets=True)
def dashboard():

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown('https://github.com/FoubaDev')


@st.cache_resource(experimental_allow_widgets=True)
def camembert():
    st.title('Proportion de valeurs du target')
   
    grouped_data = df["diagnosis"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(grouped_data, labels=grouped_data.index, autopct='%.1f')
    fig.set_size_inches (6, 6)

    
    st.pyplot(fig)
    
    
    
def main():

    with st.sidebar:
        selected = option_menu ("Diagnostic Medical Prediction System",
                            
                            ['Home',
                             'Visualization',
                            'Dataset',
                            'Prediction',
                            'Authors',
                            ],
                            icons = ['house','bi bi-file-medical-fill','person','book','person'],
                            default_index=0
    
                            )
    
    if(selected == "Home") :
        st.title("Medical Diagnostic Prediction System")
        st.title("Let's defeat breast cancer together")
        st.write("Ceci est un projet de groupe. \n\nDescription :  \n\nDévelopper un modèle d’arbre de décision pour diagnostiquer des affections médicales basées sur lessymptômes du patient et son historique médical.")
        st.write("Algorithme ML : Arbre de décision")
        dashboard()

    if(selected == "Dataset") :
        st.title("Jeu de données")
        st.write(df)
        st.write(df.shape)
       
    if(selected == "Prediction") :
        
        st.title("Veuillez entrer les valeurs des caractéristiques")
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
    
    if(selected=="Visualization"):
        
        visualization()
        camembert()
        
    if(selected == "Authors") :
         
         
        st.subheader("Authors :\n 1. ADOUM AHMAT GRENE \n\n2. BOKHIT ABDOULAYE DIGUI \n\n3. LAGRE GABBA BERTRAND \n\n4. MADJINGUESSOUM BRICE  " )
   
        st.write(" CyberSecurity Students at  Chad's Virtual University. ") 
        st.write("Github link  : https://github.com/FoubaDev/Group1TP_ML.git \n")

        st.write(" It is our pleasure to see you reading our work.")
        st.write(" This practical work aims to solve a problem concerning the classification of patients with breast cancer and those without. Such a model will help hospitals in terms of decision making on new patients with the same clinical data.")
        
if __name__=='__main__':
    main()
