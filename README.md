### 1. Introduction

In today's digital age, credit cards are a common and convenient way to manage money. However, the surge in digital payments has heightened the risk of fraud, presenting a challenge for banks and customers alike. Swift detection of fraudulent transactions is crucial for credit card companies to prevent financial losses and uphold customer trust. This project's emphasis on credit card fraud detection addresses the evolving tactics of fraudsters, who continually exploit weaknesses in digital payment systems. Beyond monetary concerns, fraud compromises financial information, damages reputations, and undermines trust in online transactions, affecting individuals and businesses. Advanced technologies like machine learning are essential to build robust defenses, enhancing the overall security of the financial system for everyone involved.

### 2. About Dataset

The dataset utilized in this project comprises credit card transactions conducted by European cardholders in September 2013. It encapsulates transactions occurring over a span of two days, encompassing a total of 284,807 transactions. Within this dataset, 492 instances represent fraudulent transactions, rendering a highly imbalanced class distribution, with frauds accounting for a mere 0.172% of all transactions.

**Acknowledgments**

The dataset utilized in this project is sourced from Kaggle and can be accessed through the following link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 3. Objectives of the Project

Develop and deploy an advanced credit card fraud detection system using machine learning, specifically unsupervised learning with autoencoders. The project aims to enhance digital transaction security by swiftly identifying fraudulent patterns, preventing financial losses for credit card companies, and reinforcing customer trust in online transactions. Through sophisticated algorithms and neural networks, the system seeks to address broader implications, protecting individuals, businesses, and contributing to the overall stability of the economy. The successful implementation will yield a potent tool for detecting and mitigating fraud, establishing a foundation for a secure and resilient financial ecosystem in the digital age.

### 4. Algorithm Selection

I employed an unsupervised learning approach for anomaly detection using TensorFlow with the Keras API. The neural network architecture utilized is based on an autoencoder.

**Reason for algorithm selection:**
The choice makes sense because traditional methods that need lots of labeled fraud examples struggle with the limited and diverse fraud data we have. The unsupervised approach lets the model figure out what normal transactions look like and spot anything different as a possible fraud. This is important because fraudsters keep changing their tactics, and it's hard to keep our fraud data updated.

The autoencoder is a good fit because credit card transactions are pretty complicated. It's like a detective that's good at finding patterns in messy situations. Unlike simpler methods, the autoencoder can understand the tricky relationships in the data. It learns on its own without needing us to tell it what fraud looks like, making it adaptable to new fraud patterns.

Another advantage is that the autoencoder can cut out the unnecessary stuff in credit card transactions. Transactions sometimes have extra details that aren't helpful, and the autoencoder can focus on what matters. This not only makes it work faster but also helps it pay attention to the important parts of the data. So, by using the unsupervised autoencoder, we're tackling fraud with a system that can adapt, handle the complexity of credit card transactions, and efficiently find meaningful patterns in the data.

### 5. Files:

**creditcard** - This is a main copy of the kaggle dataset that contains all the records of transactions (fraud and legit)

**testing_data** - This dataset contains only anomalies(only fraud transactions)

**sampling_data** - This is a random sampling data

**your_pipeline.joblib** - This file contains normalization and standardisation pipeline

**autoencoder_best__weights_.hdf5** - The model best parametres are saved here

**Main_file** - This is a complete python script file that represents the working of model in real-time

### 6. Approach:
-Run the mainfile.py python script to see the script detecting anomalies in continuous data stream.

-Here I am using a csv file instead of a database or any IoT device output in order to speedup the testing of my code. If needed we can change it to a data base stream.

-I have set the retrain threshold of the model to 50 so that the model updates itself for every 50 mistakes it makes. If needed we can change the retrain threshold value based on the requirement.

-I created all the above files from EDA (exploratory data analysis) of the given data.
