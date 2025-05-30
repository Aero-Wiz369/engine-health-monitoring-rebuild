from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/FD001.txt")
    train_model(X_train, X_test, y_train, y_test)
