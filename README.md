# 🏡 California Housing Prices Prediction

## Overview

Through this data science project, we tackled the challenge of predicting housing prices in California using machine learning techniques. By leveraging a dataset containing various features such as location, housing characteristics, and demographic information, we aimed to develop models capable of accurately estimating median house values.

To explore the model in detail, visit the Colab notebook [here](https://colab.research.google.com/drive/1-0QJfACRuZeP-PPDraePU2plXuEYJt0A?usp=sharing).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Technologies
- **Python** programming language
- **Jupyter Notebook** for interactive development
- **Pandas and NumPy** for data manipulation
- **Scikit-Learn** for machine learning modeling
- **TensorFlow and Keras** for building neural networks
- **Google Colab** for running the notebook collaboratively
- **GitHub** for version control and collaboration

## Dataset

The dataset used in this project is sourced from Kaggle and includes various features such as:

- **📍 Location:** Longitude, latitude
- **🏠 Housing characteristics:** Median age of houses, total number of rooms, total number of bedrooms
- **👥 Demographic information:** Population, number of households, median income
- **🌊 Proximity to the ocean**

## Preprocessing

The preprocessing steps involve:

1. **📥 Data Loading:** The dataset is loaded using Pandas.
2. **🧾 Handling Categorical Variables:** The `ocean_proximity` feature is converted into dummy variables.
3. **🔀 Shuffling the Dataset:** Ensuring randomness to avoid ordering effects.
4. **🗑️ Dropping Null Values:** Removing rows with missing values.
5. **📊 Splitting the Dataset:** Dividing the dataset into training, validation, and test sets.

## Models

We implemented and compared several regression models, including:

### 1. 📈 Linear Regression
A straightforward model that fits a linear relationship between input features and the target variable. While simple, it can underfit if the relationship is non-linear.

### 2. 🔍 K-Nearest Neighbors (KNN)
A non-parametric method that predicts the target value by averaging the values of the k nearest neighbors. Requires careful tuning of the hyperparameter k.

### 3. 🌳 Random Forest Regression
An ensemble learning method that constructs multiple decision trees. This model combines the predictions of individual trees to improve generalization and reduce overfitting.

### 4. 🚀 Gradient Boosting Regression
An advanced ensemble technique that builds trees sequentially, where each new tree corrects errors made by the previous ones. It uses gradient descent optimization to minimize the loss function.

### 🤖 Neural Networks

#### Simple Neural Network (Normal)
A basic neural network with a few layers and a moderate number of neurons. Suitable for simpler datasets.

#### Medium-sized Neural Network
Contains additional hidden layers or neurons compared to the simple network, allowing it to capture more complex patterns.

#### Large Neural Network
A more complex network with a greater number of hidden layers and neurons. Capable of learning intricate relationships in the data but requires more computational resources.

## Results

Through this data science project, we tackled the challenge of **predicting housing prices in California** using machine learning techniques. By leveraging a dataset containing various features such as ***location, housing characteristics, and demographic information,*** we aimed to develop models capable of accurately estimating median house values.

The ensemble model, **Gradient Boosting Regressor (GBR)**, demonstrated promising performance with a mean squared error (MSE) of approximately **49,324.05** when evaluated on the test set. This indicates that, on average, **the model's predictions deviated by approximately $49,324.05 from the actual house prices**.

Additionally, we explored the predictive capabilities of a feed-forward **neural network (NNM, MNM, LNM)** which, while not as accurate as the GBR model, provided valuable insights into the complexity of the housing price prediction task.

Overall, this project **addresses a real-world problem in the housing market by providing insights into factors influencing housing prices and delivering predictive models to assist stakeholders in making informed decisions**.

Moving forward, further refinement of the models and exploration of additional features could potentially enhance prediction accuracy and contribute to more robust decision-making processes in the real estate industry.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

Fork the project.
Create your feature branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.


## Installation

To run this project, you need to have Python installed along with the following libraries:

- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Keras

1. Clone this Repository
   git clone https://github.com/yourusername/california-housing-prices.git
   cd california-housing-prices
2. Download the dataset from Kaggle:
    - Place the dataset in the data directory.
3. Run Jupyter Notebook

**can install the required libraries using pip**:

```bash
pip install pandas numpy scikit-learn tensorflow keras
