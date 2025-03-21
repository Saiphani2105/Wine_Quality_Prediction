# 🍷 Wine Quality Classification Web App  

Welcome to the **Wine Quality Classification** project! This project is an end-to-end machine learning solution designed to predict wine quality based on various physicochemical features. The application is built using **Streamlit** and deployed on **Hugging Face**.  

## 🚀 **Overview**  
- Developed a **Wine Quality Classification** web app using a complete **Machine Learning Pipeline**.  
- Utilized **Optuna** for hyperparameter tuning to select the best model.  
- Implemented **SMOTE** to handle class imbalance.  
- Deployed the app using **Streamlit** on **Hugging Face**.  

## 🌟 **Features**  
- Predict wine quality based on input parameters.  
- Compare predictions using various machine learning models.  
- User-friendly and interactive interface.  
- Real-time classification results.  

## 🛠️ **Tech Stack**  
- **Python**  
- **Pandas, NumPy** for data manipulation  
- **Matplotlib, Seaborn** for visualization  
- **Scikit-Learn** for machine learning models  
- **Optuna** for hyperparameter tuning  
- **SMOTE** for handling class imbalance  
- **Streamlit** for building the web app  
- **Hugging Face** for deployment  

---

## 📊 **Data Description**  
The dataset consists of physicochemical features of wines, including:  
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  
- **Quality (Target Variable)**  

---

## 🔎 **Approach**  

1. **Data Exploration**  
   - Conducted exploratory data analysis to gain insights.  
   - Visualized correlations and feature importance.  

2. **Data Preprocessing**  
   - Applied **StandardScaler** for normalization.  
   - Used **SMOTE** to balance the dataset.  

3. **Model Building**  
   - Built models using **KNN**, **Random Forest**, and **Decision Tree** classifiers.  
   - Tuned hyperparameters using **Optuna**.  
   - Selected the best model for deployment.  

4. **Deployment**  
   - Saved the model using **Pickle**.  
   - Developed a Streamlit web app for user interaction.  
   - Deployed the app on **Hugging Face**.  

---

## 🧑‍💻 **How to Run the Project**  

1. Clone the repository:  
    ```bash
    git clone https://github.com/Saiphani2105/Wine_Quality_Prediction.git
    cd wine-quality-classification
    ```
 

---

## 🚀 **Live Demo**  
Check out the live web app here:  
[![Hugging Face Deployment](https://huggingface.co/spaces/Phaneendrabayi/Wine_Quality)](#)  

---

## 📁 **Project Structure**  
```bash
wine-quality-classification/
│
├── data/
│   ├── WineQT.csv
│
├── models/
│   ├── best_model.pkl
│
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
```

- `data/` - Contains the dataset.  
- `models/` - Stores the best-trained model using Pickle.  
- `app.py` - Streamlit web app for wine quality prediction.  
- `model_training.py` - Code for data preprocessing, model training, hyperparameter tuning, and model selection.  
- `requirements.txt` - List of dependencies.  
- `README.md` - Project documentation.  

---

## 🙏 **Acknowledgments**  
I extend my heartfelt thanks to **Innomatics** for providing a supportive learning environment, my tutor **Saxon**, and my mentor **Lakshmi** for their invaluable guidance throughout this project.  

---


---

If you find this project useful or have any feedback, feel free to ⭐ the repository and share your thoughts!  

#MachineLearning #DataScience #Classification #Streamlit #HuggingFace #WineQualityPrediction #EDA #AI #SMOTE #Optuna #ModelDeployment #Python #MLPipeline 
