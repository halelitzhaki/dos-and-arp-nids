# Network Intrusion Detection System (NIDS) Using Machine Learning

Welcome to the **Network Intrusion Detection System (NIDS) Using Machine Learning** project. This repository contains the implementation of a NIDS leveraging machine learning algorithms to detect and classify network intrusions effectively.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Models Implemented](#models-implemented)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the current digital era, safeguarding networks against unauthorized access and attacks is paramount. Traditional intrusion detection systems often rely on predefined signatures, which may not be effective against new or evolving threats. This project aims to enhance network security by employing machine learning techniques to identify and mitigate various forms of network intrusions.

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
- **Model Training**: Implementing and training machine learning model - Random Forest.
- **Model Optimization**: Utilizing Grid Search and Cross-Validation to fine-tune model performance.
- **Performance Evaluation**: Assessing model using metrics such as Accuracy and F1 Score.

## Datasets

The project utilizes the [NSL-KDD dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), a refined version of the KDD'99 dataset, widely used for evaluating intrusion detection systems.

## Model Implemented

- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees.

## Performance

- **Random Forest**:
  - *Without Optimization*:
    - Accuracy: 97.26%
    - F1 Score: 97.29%
  - *With Grid Search and Cross-Validation*:
    - Accuracy: 99%
    - F1 Score: 99%

- **XGBoost**:
  - *With Grid Search and Cross-Validation*:
    - Accuracy: 99%
    - F1 Score: 99%

These results demonstrate that both Random Forest, when optimized, achieve high accuracy in detecting network intrusions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/halelitzhaki/dos-and-arp-nids.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd dos-and-arp-nids
   ```
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Model Training**:
   - Run the `python3 train_model.py` script to train the machine learning model.
2. **Model Evaluation**:
   - Use the `python3 inference.py` script to show the performance of the trained model.

## Acknowledgments

This project was developed as part of an academic assignment to practice Maching Learning, Cyber and Network programming concepts.


## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or further information, please contact [halelitzhaki](https://github.com/halelitzhaki).
