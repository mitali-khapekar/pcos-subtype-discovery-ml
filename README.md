# PCOS Subtype Discovery using Machine Learning

## Overview
This project focuses on identifying hidden subtypes of Polycystic Ovary Syndrome (PCOS) using Machine Learning techniques. The system analyzes clinical and biochemical patient parameters to classify patients into two major PCOS subtypes:
- Lean PCOS
- Metabolic PCOS

The project uses K-Means Clustering for subtype discovery and Random Forest Classifier for prediction. An interactive Streamlit-based web application was also developed for real-time prediction and result visualization.

---

## Features
- Lean PCOS Prediction
- Metabolic PCOS Prediction
- K-Means Clustering
- PCA Dimensionality Reduction
- Silhouette Score Evaluation
- Streamlit User Interface
- Input Validation System

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Google Colab

---

## Machine Learning Techniques
- K-Means Clustering
- PCA (Principal Component Analysis)
- Random Forest Classifier

---

## Workflow
1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Data Normalization  
5. PCA Dimensionality Reduction  
6. K-Means Clustering  
7. Silhouette Score Evaluation  
8. Random Forest Prediction  
9. Result Display using Streamlit  

---

## Input Parameters
The system takes the following patient details as input:
- Age
- BMI
- Weight
- TSH Level
- AMH Level
- Cycle Length
- Follicle Count (Left)
- Follicle Count (Right)

---

## Screenshots

### Input UI
![Input UI](input_ui.PNG)

---

### Lean PCOS Prediction
![Lean PCOS](lean_pcos.PNG)

---

### Metabolic PCOS Prediction
![Metabolic PCOS](metabolic_pcos.PNG)

---

### Silhouette Plot
![Silhouette Plot](silhouette_plot.png)

---

### Workflow Diagram
![Workflow Diagram](workflow_diagram.png)

---

## Results
The clustering model successfully identified two significant PCOS subtypes:
- Lean PCOS
- Metabolic PCOS

The average silhouette score obtained was:

### 0.1623

This indicates a meaningful clustering structure and reasonable separation between the discovered PCOS subtypes.

---

## Future Scope
- Integration with real-time healthcare systems
- Use of larger clinical datasets
- Deep Learning based subtype prediction
- Cloud deployment for public accessibility
- Personalized treatment recommendation system

---

## Conclusion
This project demonstrates the potential of Machine Learning in healthcare analytics by identifying hidden patterns within PCOS patient data. The developed system can assist healthcare professionals in understanding PCOS heterogeneity and support personalized treatment approaches.

```
