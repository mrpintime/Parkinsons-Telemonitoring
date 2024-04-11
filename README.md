# Parkinsons-Telemonitoring Analysis
![image info](./assets/banner_2.jpeg)

# Project Overview

We try to measure Parkinson's Disease Progression by Noninvasive Speech Tests and use that data and data science approaches to analyse and predict score of the Parkinson's Disease Progression.

# Table of Contents
- [Usage](#usage)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Results](#results)
- [Business Applications](#potential-business-and-practical-applications)
- [Next Steps](#next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

# Usage
- You can easily clone the project and see the main jupyter file on your local machine by using following command:

```bash
# Cloning
git clone git@github.com:mrpintime/Parkinsons-Telemonitoring.git
```
or

```bash 
git clone https://github.com/mrpintime/Parkinsons-Telemonitoring.git
```

# Data Description

## Dataset Description

The dataset, known as the "Oxford Parkinson's Disease Telemonitoring Dataset," is a collection of biomedical voice measurements obtained from a cohort of 42 individuals diagnosed with early-stage Parkinson's disease. These individuals participated in a six-month trial involving the use of a telemonitoring device for remote monitoring of symptom progression. The voice recordings were automatically captured in the patients' homes.

The dataset consists of various columns, including the subject number, subject age, subject gender, time interval from the baseline recruitment date, motor UPDRS score, total UPDRS score, and 16 biomedical voice measures. Each row corresponds to a single voice recording, resulting in a total of 5,875 recordings across all individuals. The primary objective of the dataset is to predict the motor and total UPDRS scores ('motor_UPDRS' and 'total_UPDRS') based on the 16 voice measures.

The data is provided in ASCII CSV format. The columns of the CSV file contain information such as the subject number, age, gender, time since recruitment, motor UPDRS score, total UPDRS score, and various voice measurement features. The dataset contains approximately 200 recordings per patient, with each patient's subject number identified in the first column.

Further details about the dataset can be found on the provided link.

## Features of the Dataset

- subject#: An integer uniquely identifying each subject.
- age: Age of the subject.
- sex: Gender of the subject, where '0' represents male and '1' represents female.
- test_time: Time elapsed since recruitment into the trial, with the integer part indicating the number of days.
- motor_UPDRS: Clinician's motor UPDRS score, linearly interpolated.
- total_UPDRS: Clinician's total UPDRS score, linearly interpolated.
- Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP: Various measures of variation in fundamental frequency.
- Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA: Various measures of variation in amplitude.
- NHR, HNR: Two measures of the ratio of noise to tonal components in the voice.
- RPDE: A nonlinear dynamical complexity measure.
- DFA: Signal fractal scaling exponent.
- PPE: A nonlinear measure of fundamental frequency variation.

## Problem Statement

The aim of this study is to measure Parkinson's Disease Progression through Noninvasive Speech Tests. The dataset, comprising 19 feature columns (excluding the Subject ID column), is utilized to explore the relationship between these features and the target features, namely motor_UPDRS and total_UPDRS.

# Methodology

## Data Preprocessing

1. **Data Cleaning:** We began by removing `wrong values`, `missing value`, `rename` and `reorder columns` in our dataset. This step ensured the integrity and consistency of our dataset.

2. **Outlier Detection:** We tried to recognize outliers in dataset using `whiskers box` plot and `DBscan` clustering Techniques, it will helps us to better capture underlying pattern. 

3. **Target Variable:** We choose one of `motor_UPDRS` as target variable as it has strong positive correlation with `total_UPDRS`.

## Feature Extraction

1. **Statistical Features:** We extracted statistical features like mean, median, standard deviation, and skewness from cleaned data, These features provide a basic understanding of the distribution and variability of the features.

2. **Correlation:** We Extract Pearson correlation Coeffitient for each pair of features. These features provide a basic understanding of the relation of features based of variation.

3. **Feature Engineering:** We used techniques like Embedding, Deep Learning and Clustering to create new features and extract useful ones, this will help us to create new features which help us to capture target daat pattern which was not capture by original features.  


> The combination of these preprocessing and feature extraction techniques was critical in preparing the dataset for subsequent machine learning models. They allowed us to capture the essential characteristics of the dataset relevant for performing predictive analysis.

## Machine Learning Techniques and Algorithms

In this project, we employed a linear regression as a baseline model and we use DNN model to predict value of target variable. The following is an overview of the key algorithm and approaches we used:

### 1. Supervised Learning Algorithms

- **Linear Regression:** We use this simple model to see how well will be our crafted datasets and we probably get better results on which dataset, because of the complex dataset and task we need to extract new features and create new datasets therefore we need to evaluate these datasets with a baseline model then feed best one to DNN model.


### 2. Unsupervised Learning for Feature Learning

- We riched our project by using **Autoencoders** and Machine Learning clustering method like **DBScan**, we used them to create new stable features and recognizing outliers and noise.  

### 3. Model Evaluation and Selection

- **Performance Metrics:** `R2 Score` and `Mean Square Error` were the primary metrics used to evaluate the model. Given the complex nature of data, we focused extensively on maximizing R2Score and minimizing MSE, we also perform cross validation on 10-folds of dataset. 

# Results

My project's exploration into Parkinsons-Telemonitoring Analysis using various DNN Model has led to some noteworthy insights and conclusions. Here are the summarized results and our interpretations of cross validation:

**Note:** You can find Visual Results in notebook file in `Cross Validation` section, but i will add them here also. `(Future Version)`

| Fold |      Loss       |       Accuracy        |
|------|-----------------|-----------------------|
|   1  | 8.305935859680176 | 86.68670654296875%   |
|   2  | 6.538992404937744 | 89.63007926940918%   |
|   3  | 12.072035789489746 | 82.17684030532837%  |
|   4  | 9.777301788330078 | 84.13716554641724%   |
|   5  | 10.091413497924805 | 85.05303263664246%  |
|   6  | 8.512579917907715 | 88.44175934791565%   |
|   7  | 6.636654376983643 | 90.1929259300232%    |
|   8  | 7.845546245574951 | 87.8102958202362%    |
|   9  | 12.212318420410156 | 82.01137185096741%  |
|  10  | 6.949376583099365 | 89.9867057800293%    |

Average scores for all folds:
- Accuracy: 86.61268830299377% (±2.9580667041855335)
- Loss: 8.894215488433838 (±1.983405040425927)

### Key Insights

The cross-validation results provide valuable insights into the performance of our models. We observe that the average accuracy across all folds is approximately 86.61%, with a standard deviation of ±2.96%. Additionally, the average loss is approximately 8.89, with a standard deviation of ±1.98. These metrics indicate that our models perform consistently well in predicting the motor UPDRS score based on the provided voice measures.

Moreover, examining the individual fold results reveals variations in model performance across different subsets of the data. For instance, while some folds achieve higher accuracies exceeding 89%, others show slightly lower performance around 82%. Such variations underscore the importance of robust model evaluation and the potential influence of data distribution on model outcomes.

### Potential Business and Practical Applications

The insights derived from our analysis hold significant implications for both the healthcare sector and technological advancements:

1. **Disease Progression Monitoring:** By accurately predicting motor and total UPDRS scores through noninvasive speech tests, our models offer a practical approach for remote monitoring of Parkinson's disease progression. Healthcare providers can leverage these predictions to track patients' symptoms over time, enabling timely intervention and personalized treatment plans.

2. **Telemedicine and Remote Patient Monitoring:** The utilization of telemonitoring devices for data collection aligns with the growing trend of telemedicine. Our models facilitate remote patient monitoring, allowing individuals with Parkinson's disease to receive continuous care and support from the comfort of their homes. This not only enhances patient convenience but also reduces the burden on healthcare facilities.

3. **Early Detection and Intervention:** Early detection of Parkinson's disease progression is crucial for initiating appropriate interventions and improving patient outcomes. By identifying subtle changes in voice patterns indicative of disease progression, our models contribute to early diagnosis and intervention strategies, potentially enhancing the effectiveness of treatment regimens.

4. **Research and Development:** The insights gained from our analysis pave the way for further research and development in the field of Parkinson's disease monitoring. Researchers can explore additional voice biomarkers and advanced machine learning techniques to enhance prediction accuracy and uncover novel insights into disease progression mechanisms.

### Next Steps

Moving forward, several avenues for research and improvement can be pursued to enhance the efficacy and applicability of our models:

1. **Feature Engineering Refinement:** Continuously refining feature engineering techniques can help extract more informative features from the voice data. Exploring advanced feature selection algorithms and domain-specific knowledge integration may further enhance model performance.

2. **Model Optimization:** Fine-tuning hyperparameters and exploring alternative model architectures, such as ensemble learning and deep learning variations, can potentially improve prediction accuracy and robustness.

3. **External Validation:** Conducting external validation studies using independent datasets can validate the generalizability and reliability of our models across diverse patient populations and data sources.

4. **Clinical Integration:** Collaborating with healthcare professionals to integrate our models into clinical practice workflows is essential for real-world implementation. This involves addressing regulatory compliance, data privacy concerns, and user interface design for seamless integration into existing healthcare systems.

5. **Longitudinal Studies:** Performing longitudinal studies to track disease progression in individual patients over extended periods can provide valuable insights into the predictive power and stability of our models over time.

By pursuing these next steps, we aim to translate our research findings into tangible benefits for patients, caregivers, and healthcare providers, ultimately contributing to improved management and treatment outcomes for Parkinson's disease.

# Contributing

## Contributing to Parkinsons-Telemonitoring Analysis

I highly appreciate contributions and are excited to collaborate with the community on this data science project. Whether it's through data analysis, model improvement, documentation, or reporting issues, your input is valuable. Here’s how you can contribute:

1. **Fork the Repository:** Start by forking the project repository to your GitHub account. This creates a personal copy for you to work on.

2. **Clone the Forked Repository:** Clone the repository to your local machine. This allows you to make changes and test them locally.

   ```git
   git clone https://github.com/mrpintime/Parkinsons-Telemonitoring.git
   ```

3. **Create a New Branch:** Create a new branch for your work. This keeps your changes organized and separate from the main branch.

   ```git
   git checkout -b feature-or-fix-branch-name
   ```

4. **Contribute Your Changes:**
   - **Data Analysis:** If you’re adding new analysis, ensure your code is well-documented and follows the project’s coding conventions. Include comments and README updates explaining your methodology.
   - **Model Improvement:** For changes to existing models, provide a clear explanation and any performance metrics or results to support the improvements.
   - **Data Contribution:** If contributing new data, ensure it is properly cleaned, formatted, and accompanied by a source description.

5. **Commit and Push Your Changes:** Commit your changes with a clear message describing the update. Push the changes to your forked repository.

   ```git
   git commit -m "Detailed description of changes"
   git push origin feature-or-fix-branch-name
   ```

6. **Create a Pull Request:** Go to your fork on GitHub and initiate a pull request. Fill out the PR template with all necessary details.

7. **Code Review and Discussion:** Wait for the project maintainer(that's me 😁 ) to review your PR. Be open to discussion and make any required updates.

### Reporting Issues

- Use the Issues tab to report problems or suggest enhancements.
- Be as specific as possible in your report. Include steps to reproduce the issue, along with any relevant data, code snippets, or error messages.

### General Guidelines

- Adhere to the project's coding and data handling standards.
- Update documentation and test cases for substantial changes.
- Keep your submissions focused and relevant to the project's goals.

Your contributions play a vital role in the success and improvement of Parkinsons-Telemonitoring Analysis. We look forward to your innovative ideas and collaborative efforts!

# License
- This project is licensed under the MIT License - see the `LICENSE.md` file for details.

# Contact
- Contact me through my Linkedin: [@moein-zeidanlou](https://www.linkedin.com/in/moein-zeidanlou)

# Acknowledgments
- Credits to UCI Machine Learning Repository for providing the dataset.  
Dataset Link: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
- Code Reference for K-Fold: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
