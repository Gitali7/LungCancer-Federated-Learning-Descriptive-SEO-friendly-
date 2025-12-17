import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_utils import prepare_data, unzip_dataset
from src.model import create_resnet_model
from src.federated import train_client_local, average_weights, get_model_weights, set_model_weights, evaluate_model

# Constants
NUM_CLIENTS = 3
ROUNDS = 10
LOCAL_EPOCHS = 5
ZIP_PATH = "data/raw/archive.zip"

def main():
    print("="*50)
    print("Lung Cancer Detection - Federated Learning Simulation")
    print("="*50)
    
    # 1. Setup Data
    extracted_dir = os.path.join("data/raw", "LIDC-IDRI-slices")
    if os.path.exists(extracted_dir):
        print(f"Data already extracted at {extracted_dir}. Skipping unzip.")
    elif os.path.exists(ZIP_PATH):
        print("Dataset zip found. Unzipping...")
        unzip_dataset(ZIP_PATH)
    
    print("Preparing and splitting data...")
    # Limiting to 600 samples (200 per client approx) for clear demo
    clients_data = prepare_data(num_clients=NUM_CLIENTS, max_samples=600)
    
    if clients_data is None:
        print("[ERROR] Data preparation failed. Please ensure images are in data/raw.")
        return

    # 2. Initialize Global Model
    print("\nInitializing Global Model...")
    global_model = create_resnet_model()
    global_weights = get_model_weights(global_model)
    
    # Metrics tracking
    global_accuracies = []
    
    # 3. Federated Training Loop
    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")
        
        local_weights_list = []
        client_accuracies = []
        
        # Simulate each client
        for i in range(NUM_CLIENTS):
            X_train, y_train, X_test, y_test = clients_data[i]
            
            # Client trains locally
            print(f"  Client {i+1} training...", end="\r")
            new_weights, acc = train_client_local(X_train, y_train, global_weights, epochs=LOCAL_EPOCHS)
            
            local_weights_list.append(new_weights)
            client_accuracies.append(acc)
            print(f"  Client {i+1} completed. Local Acc: {acc:.4f}")
            
        # Aggregation Phase (Server)
        print("  Aggregating weights...")
        global_weights = average_weights(local_weights_list)
        
        # Update Global Model
        set_model_weights(global_model, global_weights)
        
        # Evaluate Global Model (Optional: Use a held-out global test set, or avg of local test sets)
        # Here we evaluate on each client's test set to see how the global model performs generally
        test_accs = []
        for i in range(NUM_CLIENTS):
            _, _, _, y_test = clients_data[i] # Just unpacking
            X_test = clients_data[i][2] # X_test
            y_test = clients_data[i][3] # y_test
            
            _, acc, _ = evaluate_model(global_model, X_test, y_test)
            test_accs.append(acc)
            
        avg_global_acc = np.mean(test_accs)
        global_accuracies.append(avg_global_acc)
        print(f"  >>> Round {round_num} Global Avg Accuracy: {avg_global_acc:.4f}")

    # 4. Final Results
    print("\n" + "="*50)
    print("Training Complete.")
    print(f"Final Global Accuracy: {global_accuracies[-1]:.4f}")
    
    # Save Model for Web App
    global_model.save('final_model.h5')
    print("Model saved to final_model.h5")
    
    # Plotting
    plt.figure()
    plt.plot(range(1, ROUNDS + 1), global_accuracies, marker='o')
    plt.title('Federated Learning Performance')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('logs/fl_accuracy_plot.png')
    print("Accuracy plot saved to logs/fl_accuracy_plot.png")

if __name__ == "__main__":
    main()
