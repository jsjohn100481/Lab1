import json
from ml_code.data_load import load_data
from ml_code.pre_processing import preprocess_data
from ml_code.models import ModelFactory
from ml_code.train import train_and_evaluate

with open('C:/Users/jsjohn/UMD/LabSession/Lab1/Test/config.json') as config_file:
    config = json.load(config_file)
    
data = load_data("C:/Users/jsjohn/UMD/LabSession/data/data.csv")
print(data.head())

X_train, X_test, y_train, y_test = preprocess_data(data)

model = ModelFactory.get_model(config["model_type"])

accuracy, cm, y_test, y_prob = train_and_evaluate(model, X_train, X_test, y_train, y_test)

#print_metrics(accuracy, cm, y_test, y_prob)