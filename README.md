# Disaster Detection From Tweets using ML 🌐🔍📜

## 📝 Description 
This project aims to detect whether a given tweet is related to a disaster or not. The goal is to build a machine learning model that can classify tweets into two categories: "Disaster" and "Normal." The model will be trained on a dataset of tweets that are labeled as disaster-related or not.

## 🎯 Dataset 
The dataset used for this project contains labeled tweets, where each tweet is marked as either a disaster tweet or a normal tweet. The dataset will be split into training and testing sets for model training and evaluation.

## ⚙️ Approach 
1. Import Dependencies: Install and import the required Python libraries for data preprocessing, machine learning, and evaluation.
2. Load Dataset: Load the dataset containing labeled tweets for training and testing the model.
3. Data Cleaning: Clean the text data by removing stopwords, punctuation, and non-alphabetic characters.
4. Feature Extraction: Use TF-IDF vectorization to convert text data into numerical features suitable for machine learning.
5. Model Selection: Experiment with different classifiers such as Multinomial Naive Bayes and Passive Aggressive Classifier.
6. Train Model: Train the selected classifiers on the TF-IDF transformed training data.
7. Model Evaluation: Evaluate the performance of each model using metrics like accuracy, precision, recall, specificity, and F1 score.
8. Most Informative Features: Identify the most informative features used by the classifiers for classification.
9. Sample Prediction: Make predictions on sample tweets to see the model's performance on new data.

## 📥 Installations 
To run the code, you need to install the required Python libraries. Use the following command to install the dependencies:

```
pip install pandas numpy nltk seaborn scikit-learn matplotlib wordcloud
```

## ⚙️ Setup 
1. Clone the repository and navigate to the project directory.
2. Make sure you have the required Python libraries installed.
3. Execute the "code-Notebook" script to load the dataset, preprocess the data, train the models, and make predictions.

## 🛠️ Requirements 
- Python 3.x
- Pandas
- NumPy
- NLTK
- Seaborn
- Scikit-learn
- Matplotlib
- WordCloud

## 🚀 Technology Used 
- Python
- Natural Language Processing (NLP)
- Text Data Preprocessing
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Multinomial Naive Bayes Classifier
- Passive Aggressive Classifier

## ▶️ Run 
1. After setting up the environment and dependencies, run the "code-Notebook".
2. The script will load the dataset, preprocess the text data, extract features using TF-IDF, train the classifiers, and evaluate their performance.
3. The most informative features used by the classifiers will be displayed.
4. Sample tweets will be predicted by the model to check its performance on new data.

---

📝 I frequently create content focused on complete projects in the realm of data science using Python and R.

💬 Feel free to inquire about data science, computer vision, Python, and R.

---

<p align="Right">(◕‿◕) Thank you for exploring my GitHub project repository. 👋</p>
