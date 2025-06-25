# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY NAME:CODTECH IT SOLUTION

INTERN NAME:JUNAIDA ADAKANDY MOIDU

DOMAIN NAME:MACHINE LEARNING

MENTOR NAME:NEELA SANTHOSH

This project focuses on implementing Sentiment Analysis on customer review data using techniques from Natural Language Processing (NLP) and Machine Learning (ML). The main objective is to predict the sentiment behind each customer review, classifying it as either positive or negative based on the textual content.

In today’s digital world, millions of customers express their opinions about products and services online. Understanding customer sentiment helps businesses make data-driven decisions, improve their products, address customer concerns proactively, and enhance overall user experience. This project showcases how NLP techniques combined with machine learning models can automate the process of identifying customer sentiments at scale.

The task was performed as part of my Machine Learning Virtual Internship Program at CodeTech IT Solutions, under the topic "Natural Language Processing with Sentiment Classification."

The solution focuses on end-to-end steps starting from data loading, text preprocessing, vectorization using TF-IDF, applying a Logistic Regression classifier, evaluating the model, and testing on custom inputs.

 Tools and Technologies Used:
 
For developing this project, I have used Python programming language due to its strong support for machine learning and data processing libraries. Below is the list of tools and libraries used:

Python 3.x: Main programming language used for implementation.

Jupyter Notebook: Integrated Development Environment (IDE) for writing and executing Python code with rich text and visualization support.

Pandas: Used for data loading, data manipulation, and cleaning operations.

NumPy: Used for numerical operations and handling arrays efficiently.

Scikit-learn (sklearn): Provided modules for TF-IDF vectorization, machine learning model training (Logistic Regression), and evaluation metrics.

Matplotlib and Seaborn: Used for data visualization, especially for plotting the confusion matrix for model performance evaluation.

Dataset Used:
The dataset used for this project is the Amazon Fine Food Reviews dataset, publicly available on Kaggle.

Dataset Name: Amazon Fine Food Reviews

Source: Kaggle Dataset

Description:

The dataset contains over 500,000 reviews from Amazon customers about food products. Each review includes multiple fields such as the review text, customer ID, product ID, rating score (1 to 5), and review summary.

For this task, I mainly focused on two columns:

Text – Contains the detailed customer review content.

Score – Numerical rating given by the customer (on a scale from 1 to 5).

Neutral reviews (where the score equals 3) were removed from the dataset to maintain a clear distinction between positive and negative sentiments.

Project Workflow and Methodology:

Data Loading and Cleaning:

Loaded the CSV dataset using Pandas.

Checked for missing values and removed rows with null entries.

Filtered out neutral reviews.

Converted rating scores into binary sentiment labels: Positive (1) for scores above 3 and Negative (0) for scores below 3.

Text Preprocessing and Vectorization:

Applied TF-IDF Vectorization (Term Frequency-Inverse Document Frequency) to convert the text data into numerical feature vectors.

Limited the TF-IDF feature space to the top 10,000 most important tokens using unigrams and bigrams.

Removed English stopwords to reduce noise in the feature space.

Model Building:

Used Logistic Regression, a widely used linear classification algorithm from Scikit-learn, suitable for binary classification tasks.

Trained the model using the TF-IDF-transformed training data.

Model Evaluation:

Evaluated the model’s performance on test data using classification metrics such as precision, recall, F1-score, and support.

Generated and visualized the confusion matrix to see the true positives, true negatives, false positives, and false negatives.

Calculated model accuracy and compared predicted labels with actual labels.

Custom Sentiment Prediction:

Added a custom function to test the model on new, user-defined review texts.

Tested with examples like "Excellent product, very satisfied!" and "Terrible quality, waste of money."

 Results and Observations:
 
The Logistic Regression model achieved satisfactory results on test data. The classification report showed good precision and recall scores for both positive and negative classes. The confusion matrix visualization helped to easily observe how well the model distinguished between the two sentiment categories. The model was also able to predict sentiments for new, unseen review texts, demonstrating its ability to generalize well.

 Real-world Applications:
Sentiment Analysis models like this one have wide-ranging applications in the real world, including but not limited to:

Customer Feedback Analysis

Product Rating Predictions

Brand Reputation Monitoring

Social Media Sentiment Tracking

Market Research Surveys

Review Moderation and Filtering

Such models help businesses and organizations automate the process of analyzing customer opinions across large datasets, saving both time and effort.

 Conclusion:
This project provided hands-on experience in implementing a complete NLP-based sentiment classification pipeline. It strengthened my understanding of text preprocessing, TF-IDF feature engineering, supervised learning with Logistic Regression, and model evaluation techniques. Through this internship task, I gained practical knowledge of how machine learning can be used to solve real-world text classification problems, which will be valuable for future data science and machine learning projects.

![Image](https://github.com/user-attachments/assets/95d788bd-ae02-432f-b975-c522be18360a)

![Image](https://github.com/user-attachments/assets/49eeef62-c279-46b0-ad85-9338978f1fe2)

![Image](https://github.com/user-attachments/assets/6f9eb054-6a29-4fcf-a72a-1e2229dfa382)

![Image](https://github.com/user-attachments/assets/895c0d8c-47eb-4e37-a6f6-712a18b8d7a8)

![Image](https://github.com/user-attachments/assets/9f17804e-5ad3-49c6-93a2-875b408deb8d)

 









