# Mildew Detection in Cherry Leaves

## Project Overview

This project addresses a critical quality control issue for *Farmy & Foods*, a fictional agricultural company that specializes in cherry production. The company's cherry plantations are experiencing cases of **powdery mildew**, a fungal disease that affects the leaves and degrades product quality.

The current manual inspection process is slow, costly, and not scalable. This project aims to:
- **Visually differentiate** healthy and infected leaves.
- **Predict mildew presence** using image classification.
- **Deploy an interactive dashboard** for image upload and prediction.

By automating the detection process, *Farmy & Foods* can save inspection time, reduce operational costs, and protect its product quality.

## CRISP-DM Methodology

| Phase | Description |
|-------|-------------|
| **Business Understanding** | Predict powdery mildew on cherry leaves from image data to reduce manual labor and improve quality control. |
| **Data Understanding** | Image dataset of healthy and infected cherry leaves (256x256 pixels). Class distribution and image quality evaluated. |
| **Data Preparation** | Images resized, normalized, and split into train/validation/test sets. Data augmentation used for generalization. |
| **Modeling** | CNN model trained using TensorFlow/Keras. Optimized via dropout, early stopping, and learning rate tuning. |
| **Evaluation** | Confusion matrix, accuracy, and learning curves. Success defined as ≥97% accuracy on test data. |
| **Deployment** | Streamlit dashboard built for end-user image prediction, hosted via Heroku. |

## Business Requirements

### Requirement 1: Visual Differentiation Study
- Show how healthy and infected leaves differ using:
  - **Average images**
  - **Variability maps**
  - **Montage of sample images**
- Use conventional EDA techniques to support analysis.

### Requirement 2: Predictive ML Model
- Build a neural network to classify cherry leaf images.
- Prediction output: Healthy or Powdery Mildew with confidence score.
- Model must reach at least **97% accuracy** on test set.

### Requirement 3: Deployment
- Create an interactive dashboard for:
  - Uploading images
  - Viewing predictions and confidence
  - Downloading result tables
- Provide UI feedback on model performance and hypothesis validation.

## User Stories

### Epic 1: Data Understanding and Visualization
- *As a data analyst*, I want to study the visual differences between healthy and mildew-infected leaves so I can better understand the problem.

### Epic 2: Predictive Model Development
- *As a data scientist*, I want to train a neural network that predicts mildew so that the company can automate the inspection process.

### Epic 3: Dashboard for Business Use
- *As an IT user*, I want to upload leaf images and receive instant predictions so I can act quickly in the field.

## ML Task & Business Case

| Item | Description |
|------|-------------|
| **ML Task** | Binary image classification |
| **Learning Method** | Supervised Learning – Convolutional Neural Network |
| **Objective** | Predict if a cherry leaf is healthy or has powdery mildew |
| **Ideal Outcome** | ≥97% test accuracy |
| **Model Output** | Class label (Healthy / Powdery Mildew) + probability |
| **User Relevance** | Enables scalable, fast, and accurate detection |
| **Heuristics** | Image size: 50×50 or 100×100; use augmentation, dropout |
| **Training Data** | Cherry leaves dataset from Kaggle |

## Dashboard Pages & Design

### 1. **Project Summary**
- Text overview of project context and objectives
- Dataset summary and business requirements
- Widget: Informational markdown

### 2. **Visual Differentiation Study**
- Image montage (healthy vs infected)
- Average and variability images per class
- Difference between average class images
- Scatterplot of image dimension distribution
- Histogram of pixel intensity
- Widgets: Checkboxes, sliders, selectbox, button for generating image montage
- Interpretation markdown and Streamlit `st.info`, `st.warning` for user guidance

### 3. **Mildew Detection Tool**
- Image file uploader (multiple image support)
- Inline preview of each uploaded image
- Real-time class prediction with confidence score
- Table of results (with download option)
- Widgets: File uploader, prediction table, download button

### 4. **Hypothesis & Validation**
- Clearly stated hypothesis
- Validation through visual inspection & model results
- Qualitative and quantitative evaluation summary
- Widget: Informational markdown and structured layout

### 5. **Model Performance**
- Learning curves (accuracy/loss)
- Confusion matrix and class distribution
- Test metrics: accuracy, precision, recall, F1
- Statement whether model achieved required performance

## Model Evaluation

- Loss/accuracy plot (train & validation)
- Confusion matrix
- Accuracy, precision, recall, F1-score
- Statement of whether model met business goals

## Technologies Used

| Tool | Purpose |
|------|--------|
| Python | Core language |
| pandas, numpy | Data handling |
| matplotlib, seaborn, plotly | Visualizations |
| Pillow | Image processing |
| scikit-learn | Metrics & pipeline utilities |
| TensorFlow/Keras | CNN model building |
| Streamlit | Dashboard UI |
| GitHub + Git | Version control |
| Heroku | Deployment platform |

## Deployment Checklist

- `requirements.txt` with all dependencies
- `Procfile`, `setup.sh`, `runtime.txt` for Heroku deployment
- Modularized code in `src/` and `app_pages/`
- Model file saved for loading predictions
- Dashboard built with Streamlit and page navigation

## Ethical & Privacy Considerations

- Data is under NDA and must not be shared.
- This repository is for educational/demo purposes.
- Model is intended for aiding professionals, not replacing expertise.

## Version Control

This project is version-controlled using **Git & GitHub**. Commit messages follow a clear format and branches are used to separate experimentation from stable code.

## Dataset Content

The dataset comprises **cherry leaf images** from real plantations, labeled as either **healthy** or infected with **powdery mildew**, a common fungal disease. These images are standardized at 256×256 pixels and were collected by the IT & Innovation team at Farmy & Foods for the purpose of building a computer vision solution.


## Business Requirements

As a Data Analyst working on behalf of Farmy & Foods, your role is to provide insights and a deployable solution to streamline disease detection in cherry leaves. Currently, the company is facing challenges with timely identification of mildew in large-scale plantations.

* **Requirement 1:** The client seeks a visual study that helps differentiate healthy cherry leaves from those infected with powdery mildew.
* **Requirement 2:** The client wants an automated prediction tool that identifies whether a leaf is healthy or diseased from an uploaded image.


## Hypothesis and Validation

We hypothesize that:
- Cherry leaves affected by mildew present **distinct visual signs**, such as discoloration, texture changes, or fungal marks, which can be learned by a neural network.
- These differences can be revealed via average image studies, variability analysis, and class montages.

To validate:
- We will generate average and variability images.
- We will compare image statistics between healthy and mildew-infected classes.
- We will train and evaluate a deep learning model.


## Rationale: Business Requirements → Visualizations & ML Tasks

### Business Requirement 1: Visual Differentiation Study
- Display **mean and standard deviation images** per class.
- Show the **difference between average healthy and infected images**.
- Present **image montages** for each class to illustrate sample diversity.
- Plot image dimension distribution and pixel intensity histogram.

### Business Requirement 2: Image Classification
- Build a binary image classifier (CNN) to distinguish between classes.
- Provide **real-time predictions** via a Streamlit dashboard.
- Include **model performance evaluation** (confusion matrix, accuracy, confidence levels).

## Dashboard Design (Streamlit App UI)

### Page 1: Project Summary
- Overview of project goals and background.
- Details about dataset origin and structure.
- Clarification of business requirements and relevance.

### Page 2: Leaf Visualizer (Visual Differentiation Study)
- For answering Business Requirement 1
  - Show average & variability image differences.
  - Highlight key visual signs using montages and overlays.
  - Show dimension distribution & intensity histogram.

### Page 3: Mildew Detector (Prediction Interface)
- For answering Business Requirement 2
  - Upload multiple cherry leaf images.
  - Display prediction label and probability for each image.
  - Show prediction table + download option.

### Page 4: Hypothesis & Validation
- Clearly list hypotheses.
- Describe validation method and conclusions drawn from modeling and EDA.

### Page 5: ML Performance
- Visualize class balance across splits.
- Show model training history (accuracy/loss curves).
- Provide test performance metrics and interpretation.

## Deployment

The app is deployed on Heroku using:
- **Procfile** for command execution
- **setup.sh** for installing system packages (e.g., Streamlit)
- **runtime.txt** to specify Python version
- **requirements.txt** for Python package dependencies

> The deployed app supports image uploads and real-time prediction using the trained CNN model.

## Main Data Analysis and Machine Learning Libraries

- `pandas`, `numpy`: Data wrangling and preprocessing
- `matplotlib`, `seaborn`, `plotly`: Visual analytics and plots
- `Pillow`, `matplotlib.image`, `tensorflow.keras.preprocessing.image`: Image reading & transformation
- `tensorflow`, `keras`: Model definition and training
- `joblib`: Save/load Python objects like models and metadata
- `scikit-learn`: Evaluation metrics
- `streamlit`: Interactive dashboard development

## Credits

- **Content:** All website content and logic were developed by Robert Divkovic.
- **Code Institute**: Project structure inspiration and dataset curation.
- **Kaggle**: Dataset source [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Stack Overflow**: For problem solving

### The Sharp-Mind project draws inspiration from Code Institute preparation full stack project: 

- [Code Institute, WalkthroughProject01 Malaria Detection Project](https://github.com/GyanShashwat1611/WalkthroughProject01)

### Acknowledgments and Special Thanks to :

### Code Institute

#### This project was developed as part of the Code Institute's Full Stack Software Development program.