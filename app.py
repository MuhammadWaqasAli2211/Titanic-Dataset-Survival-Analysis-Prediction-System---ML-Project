import streamlit as st
import numpy as np
from joblib import load
import base64

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

model = load("bagging_model.joblib")

st.sidebar.markdown("""
<style>
.sidebar-card {
    padding: 12px;
    margin-bottom: 10px;
    border-radius: 10px;
    border: 2px solid #d0d0d0;
    text-align: center;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}

.sidebar-card:hover {
    background-color: #e3f2fd;
    border-color: #1f77b4;
}

.sidebar-active {
    background-color: #1f77b4;
    color: white;
    border-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<h3 style='text-align:center; color:#1f77b4;'>Navigation</h3>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["Home", "Real-time Predict Survival", "Visualizations", "About Project"],
    label_visibility="collapsed"
)

def sidebar_card(title, is_active=False):
    css_class = "sidebar-card sidebar-active" if is_active else "sidebar-card"
    st.sidebar.markdown(
        f"<div class='{css_class}'>{title}</div>",
        unsafe_allow_html=True
    )

sidebar_card("Home", page == "Home")
sidebar_card("Real-time Predict Survival", page == "Real-time Predict Survival")
sidebar_card("Visualizations", page == "Visualizations")
sidebar_card("About Project", page == "About Project")

st.sidebar.markdown(
    "<h3 style='text-align:center; color:#1f77b4; font-size: 14px;'>Designed by: Muhammad Waqas Ali</h3>",
    unsafe_allow_html=True
)

if page == "Home":

    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>Titanic Survival Prediction</h1>
        <h3 style='text-align: center; color: gray;'>Machine Learning Project | Bagging Ensemble Model</h3>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://loveincorporated.blob.core.windows.net/contentimages/gallery/75c4d9f2-cbea-463b-bbf3-28090f383256-titanic.jpg' 
             width='500px' 
             style='border-radius: 15px;'>
    </div>
""", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style='color: #1f77b4;'>Objective:</h3>
    To enable students to handle real-world noisy data by designing, training, and comparing multiple machine learning models, and applying ensemble learning techniques to improve accuracy.
    
    <h3 style='color: #1f77b4;'>Models Used:</h3>
                           
    - Decision Tree Classifier 
    - Naïve Bayes Classifier 
    - Support Vector Machine
    - Ensemble Technique  
    """, unsafe_allow_html=True)

elif page == "Real-time Predict Survival":
    st.title("Survival Prediction")
    st.write("Fill passenger details below")

    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        age = st.slider("Age", 1, 80, 25)
        sibsp = st.number_input("Siblings / Spouses", 0, 8, 0)
        parch = st.number_input("Parents / Children", 0, 6, 0)

    with col2:
        fare = st.slider("Fare", 0.0, 500.0, 32.0)
        sex = st.radio("Gender", ["Male", "Female"])
        embarked = st.selectbox("Embarked From", ["C", "Q", "S"])
    
    sex_male = 1 if sex == "Male" else 0
    sex_female = 1 if sex == "Female" else 0
    embarked_C = 1 if embarked == "C" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0
    
    input_data = np.array([[pclass, age, sibsp, parch, fare,
                            sex_female, sex_male,
                            embarked_C, embarked_Q, embarked_S]])
    
    st.markdown("---")
    
    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("Passenger Survived ✅")
        else:
            st.error("Passenger Did NOT Survive ❌")

elif page == "Visualizations":
    st.title("Data Visualization")
    st.write("This section explains the fundamentals for understanding sequences and comparing variables, effectively revealing direction and rate of change. ")
    
    tab1, tab2, tab3 = st.tabs([
        "Feature Comparission",
        "Model Accuracy",
        "Confusion Matrix"
    ])
    
    with tab1:
        st.markdown("""
        <h3 style='color: gray;'>Correlations of Features</h3>
        """, unsafe_allow_html=True)

        import base64

        image_path1 = "assets/Correlation.png"

        with open(image_path1, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode()

        col1, col2 = st.columns([1, 1])  # Adjust ratio if you want wider image or text

        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                    <img src='data:image/png;base64,{b64_image}' 
                        style='width:100%; max-width:400px;'>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.info("""
                → Lower class number (1st class) passengers had a higher chance of survival.

                → Passengers who paid higher fares were more likely to survive, which aligns with class-based evacuation priority.

                → Age alone did not strongly determine survival, though younger passengers had a slight advantage.

                → Having some family aboard helped survival, but too many dependents reduced chances.

                ##### Overall Insight:
                No single feature strongly decides survival alone; a combination of class, fare, and family size plays an important role.
            """)

        st.markdown("""
            <h3 style='color: gray;'>Passenger Class vs Survival</h3>
        """, unsafe_allow_html=True)

        image_path2 = "assets/Countplot.png"

        with open(image_path2, "rb") as img_file:
            b64_image2 = base64.b64encode(img_file.read()).decode()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("""
                
                → First-class passengers had the highest survival rate.

                → Third-class passengers experienced the highest number of deaths.

                → Second-class survival lies between first and third class.

                #### Insight:
                ##### Survival chances were strongly influenced by passenger class, reflecting social inequality during evacuation.
            """)

        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                    <img src='data:image/png;base64,{b64_image2}' 
                        style='width:100%; max-width:400px;'>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style='color: gray;'>Feature Relationships</h3>
        """, unsafe_allow_html=True)

        image_path3 = "assets/Pairplot.png"

        with open(image_path3, "rb") as img_file:
                    b64_image3 = base64.b64encode(img_file.read()).decode()

        col1, col2 = st.columns([1, 1])

        with col1:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                            <img src='data:image/png;base64,{b64_image3}' 
                                style='width:100%; max-width:400px;'>
                        </div>
                    """, unsafe_allow_html=True)

        with col2:
                    st.info("""
                        → Survivors tend to have higher fares compared to non-survivors.

                        → Younger passengers show a slight survival advantage.

                        → Small family sizes had better survival outcomes.

                        ##### Insight:
                        Feature overlap indicates non-linear patterns, supporting the use of advanced ML models.
                    """)

        st.markdown("""
                <h3 style='color: gray;'>Passenger Gender vs Survival</h3>
            """, unsafe_allow_html=True)


        image_path4 = "assets/Countplot2.png"

        with open(image_path4, "rb") as img_file:
                b64_image4 = base64.b64encode(img_file.read()).decode()

        col1, col2 = st.columns([1, 1])

        with col1:
                st.info("""
                    
                    → Female passengers had the highest survival rate.

                    → Male passengers experienced the highest number of deaths.


                    #### Insight:
                    ##### Survival chances were strongly influenced by passenger Gender.
                """)

        with col2:
                st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                        <img src='data:image/png;base64,{b64_image4}' 
                            style='width:100%; max-width:400px;'>
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Models Accuracy")

        image_path5 = "assets/Model accuracy.png"

        with open(image_path5, "rb") as img_file:
                b64_image5 = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
                    <div style='text-align: center; border-radius:15px; border: 1px inset gray;'>
                        <img src='data:image/png;base64,{b64_image5}' 
                            style='width:80%;'>
                    </div>
                """, unsafe_allow_html=True)

        st.success("""
        - **Decision Tree:** Easy to interpret, may overfit  
        - **Naïve Bayes:** Fast, assumes independence  
        - **SVM:** Handles complex boundaries, needs scaling  

        **Bagging Ensemble:** Combines multiple Decision Trees to reduce overfitting and improve generalization
        """)
    
    with tab3:
        st.subheader("Performance Evaluation")


        image_path_DT = "assets/CM DT.png"
        image_path_NB = "assets/CM NB.png"
        image_path_SVM = "assets/CM SVM.png"
        image_path_B = "assets/CM Bagging DT.png"

        with open(image_path_DT, "rb") as img_file:
                b64_image_DT = base64.b64encode(img_file.read()).decode()

        with open(image_path_NB, "rb") as img_file:
                image_path_NB = base64.b64encode(img_file.read()).decode()

        with open(image_path_SVM, "rb") as img_file:
                image_path_SVM = base64.b64encode(img_file.read()).decode()

        with open(image_path_B, "rb") as img_file:
                image_path_B = base64.b64encode(img_file.read()).decode()

        col1, col2 = st.columns([1, 1])
        col3, col4 = st.columns([1, 1])
            
        with col1:
                st.success("""
                    **Decision Tree:**
                """)
                st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray; margin-bottom: 10px;'>
                        <img src='data:image/png;base64,{b64_image_DT}' 
                            style='width:100%; max-width:400px;'>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
                st.success("""
                    **Naïve Bayes:**
                """)
                st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                        <img src='data:image/png;base64,{image_path_NB}' 
                            style='width:100%; max-width:400px;'>
                    </div>
                """, unsafe_allow_html=True)

        with col3:
                st.success("""
                    **Support Vector Machine:**
                """)
                st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                        <img src='data:image/png;base64,{image_path_SVM}' 
                            style='width:100%; max-width:400px;'>
                    </div>
                """, unsafe_allow_html=True)

        with col4:
                st.success("""
                    **Bagging Decision Tree:**
                """)
                st.markdown(f"""
                    <div style='text-align: center; padding: 5px; border-radius:15px; border: 1px inset gray;'>
                        <img src='data:image/png;base64,{image_path_B}' 
                            style='width:100%; max-width:400px;'>
                    </div>
                """, unsafe_allow_html=True)


elif page == "About Project":
    st.title("About This Project")
    st.write("This section explains how the project was built, why models were chosen, and what we learned.")
    
    tab1, tab2, tab3 = st.tabs([
        "Data Preprocessing Decisions",
        "Model Comparison & Justification",
        "Observations & Conclusion"
    ])
    
    with tab1:
        st.subheader("Data Preprocessing Decisions")
        st.info("""
        Real-world data is rarely clean. Steps taken:

        1. **Handling Missing Data:**  
        - `fare` & `embarked` missing values removed  
        - `age` missing values filled with median based on gender & class  

        2. **Removing Irrelevant Features:**  
        - Drop columns with >60% missing (`cabin`, `body`, `boat`, `home.dest`)  
        - Drop non-predictive columns (`name`, `ticket`)  

        3. **Encoding Categorical Data:**  
        - `sex` & `embarked` converted using One-Hot Encoding  

        4. **Feature Scaling:**  
        - `age` & `fare` standardized
        """)
    
    with tab2:
        st.subheader("Model Comparison & Justification")
        st.success("""
        - **Decision Tree:** Easy to interpret, may overfit  
        - **Naïve Bayes:** Fast, assumes independence  
        - **SVM:** Handles complex boundaries, needs scaling  

        **Bagging Ensemble:** Combines multiple Decision Trees to reduce overfitting and improve generalization
        """)
    
    with tab3:
        st.subheader("Observations & Conclusion")
        st.warning("""
        **Observations:**  
        - Decision Tree overfitted (training > test accuracy)  
        - Naïve Bayes stable but slightly less accurate  
        - SVM performed well after scaling  
        - Bagging improved test accuracy & reduced variance  

        **Overfitting Control:** Bagging reduced noise and variance  

        **Conclusion:**  
        - Proper preprocessing is crucial  
        - Ensemble learning improves prediction on noisy data  
        - Bagging Decision Tree was the most reliable
        """)