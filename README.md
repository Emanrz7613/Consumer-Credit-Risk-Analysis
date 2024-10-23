# Consumer-Credit-Risk-Analysis
Predicting consumer credit risk using machine learning techniques with real-world financial data.
## Overview
This project aims to predict whether a borrower will repay a loan using deep learning models on consumer credit data. Accurate predictions of credit risk can help financial institutions make better lending decisions, reduce defaults, and mitigate financial losses.
## Motivation
In the finance industry, it is critical to assess the credit risk of individuals before issuing loans. This project leverages deep learning techniques to predict the likelihood of loan repayment or default using various consumer features. By building robust machine learning models, we can provide insights into how different factors affect the probability of loan default.
## Dataset
You can download the dataset from [this link](https://drive.google.com/file/d/1zra3u2OqUwsENAdeQqp7G3DQGfD9EF1d/view?usp=sharing)

The dataset used for this project contains **395,932 loan records** with **31 features**, collected between 2007 and 2014. Some of the important features include:
- **Loan amount**
- **Annual income**
- **Debt-to-income ratio**
- **Number of open accounts**
- **Revolving balance**
- **Employment length**
- **Target variable**: 
  - `good_bad`: 1 when the borrower repays the loan, 0 when the loan is not repaid.
## Models Used
We implemented and compared three types of deep learning models:
### 1. **Deep Neural Network (DNN)**
   - **Architecture**: A simple 3-layer network.
     - First Dense layer: 128 neurons, ReLU activation.
     - Second Dense layer: 64 neurons, ReLU activation.
     - Output layer: 1 neuron for binary classification, Sigmoid activation.
   - **Optimizer**: Adam
   - **Loss function**: Binary Cross-Entropy
   - **Evaluation**: 
     - Accuracy: 89.63%
     - Precision for loan repayment: Low
     - Recall for loan repayment: High
     - **Confusion Matrix**: 
       - True Positive: 14,664
       - True Negative: 612
       - False Positive: 1,592
       - False Negative: 290
### 2. **Multilayer Perceptron (MLP)**
   - **Architecture**: 
     - Dense layer: 64 neurons, ReLU activation.
     - Dropout layers with a rate of 0.5 to prevent overfitting.
     - Final Dense layer with 1 neuron and Sigmoid activation for binary classification.
   - **Training**:
     - Data split: 80% training, 20% testing
     - Trained for 10 epochs with a batch size of 32.
     - Validation split: 20%
   - **Results**:
     - Accuracy: 90.0%
     - Sensitivity (Recall): 99.83%
     - Specificity: 2.04%
     - Precision: 90.08%
     - **Confusion Matrix**:
       - True Positive: 71,079
       - True Negative: 163
       - False Positive: 7,826
       - False Negative: 119
### 3. **Wide & Deep Neural Network**
   - **Architecture**: 
     - Combines the benefits of memorization from a wide model and generalization from a deep model.
     - Deep component: Dense layers with ReLU activation.
     - Wide component: Linear model capturing simple patterns.
   - **Evaluation**:
     - High F1 score, indicating strong precision and recall.
     - ROC-AUC score suggests the model has good discriminative ability.
## Evaluation Metrics
For each model, we evaluated the following metrics:
- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: How often a predicted default is actually a default.
- **Recall**: Ability to detect actual defaults.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measure of model’s ability to distinguish between classes.
## Key Findings
- The **MLP model** outperformed others, achieving the highest accuracy (90.0%) and a near-perfect recall (99.83%) for detecting loan repayment.
- All models performed well, but the high number of **false positives** indicated a challenge in predicting loan defaults accurately.
- **DNN** and **Wide & Deep** models also provided strong results, especially in terms of recall, but MLP was the best overall.
## How to Run
To replicate the project, follow these steps:
1. **Clone the repository**:
    ```bash
    git clone https://github.com/username/consumer-credit-risk.git
    cd consumer-credit-risk
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the notebooks**:
    - Open Jupyter Notebook and run the following:
      - `MLP.ipynb`
      - `Deep_and_Wide_Model.ipynb`
      - `Group_4_DNN.ipynb`
4. **Explore the models**: Follow the instructions in each notebook to load the data, train models, and evaluate results.
## Future Work
- Incorporate more advanced deep learning techniques such as **LSTM** for time-series loan data.
- Fine-tune hyperparameters for better precision in predicting loan defaults.
- Handle class imbalance using techniques such as **SMOTE** or **Weighted Losses**.
## Contributors
This project was developed by:
- **Eric Chaves**
- **Kwame Sefa-Boateng**
- **Jordyn Dolly**
- **Han Zhang**
Feel free to reach out on LinkedIn or via email if you have any questions or would like to collaborate!
