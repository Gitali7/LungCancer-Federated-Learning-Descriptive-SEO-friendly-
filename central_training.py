import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import prepare_data
from src.model import create_resnet_model
from src.federated import evaluate_model
from sklearn.model_selection import train_test_split

def train_centralized():
    print("="*50)
    print("Centralized Learning Baseline")
    print("="*50)

    # Re-use prepare_data but we will merge everything back for centralized training
    # Ideally we should just load all data at once, but we can aggregate the splits
    clients_data = prepare_data(num_clients=1, max_samples=600) # Treat as 1 giant client
    
    if clients_data is None:
        print("No data found.")
        return

    X_train, y_train, X_test, y_test = clients_data[0]
    
    print(f"Centralized Data: Train={len(X_train)}, Test={len(X_test)}")
    
    model = create_resnet_model()
    
    print("Training Centralized Model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    
    loss, acc, recall = evaluate_model(model, X_test, y_test)
    print(f"Final Centralized Accuracy: {acc:.4f}")
    
    # Save plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Centralized Model Accuracy')
    plt.legend()
    plt.savefig('logs/centralized_accuracy_plot.png')
    print("Saved plot to logs/centralized_accuracy_plot.png")
    
    return acc

if __name__ == "__main__":
    train_centralized()
