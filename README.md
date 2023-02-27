# 🕵️‍♂️ Sole Trader Classifier

Sole Trader Classifier is a Python-based solution for classifying customers as either sole traders or private individuals based on their accounting data. The solution is designed to identify patterns in the customer's transaction history, business-related indicators, and contextual information to determine whether they are a sole trader or a private individual.

## 🛠️ Installation

To run the project, you will need to have Python 3.7 or later installed on your machine. You can download Python from the official website.

After installing Python, you can clone the project repository to your local machine using the following command:

```bash
git clone https://github.com/lxndralbert/sole-trader-classifier.git
```

Once the repository is cloned, you can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

To use the solution, you need to have a data file in CSV format containing the customer's accounting data. The data file should contain columns for transaction history, business-related indicators, and contextual information.

To train the Sole Trader Classifier, navigate to the project directory and run the following command:

```bash
python train.py /path/to/customer/data.csv
```

Replace /path/to/customer/data.csv with the actual path to the customer data file on your machine.

The solution will train a binary classification model using the customer data and output the accuracy of the model on the training data.

To use the trained model to classify new customers, run the following command:

```bash
python predict.py /path/to/new/customer/data.csv
```

Replace /path/to/new/customer/data.csv with the actual path to the new customer data file on your machine.

The solution will use the trained model to classify each customer in the new data file as either a sole trader or a private individual and output the results to a new file in CSV format.

## 🤝 Contributing

If you find a bug or have a feature request, please open an issue on the project repository. If you would like to contribute to the project, you can fork the repository, make your changes, and submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
