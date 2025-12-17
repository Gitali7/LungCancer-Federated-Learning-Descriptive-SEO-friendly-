import numpy as np
from src.model import create_resnet_model

def get_model_weights(model):
    """Extracts weights from a model."""
    return model.get_weights()

def set_model_weights(model, weights):
    """Sets the weights of a model."""
    model.set_weights(weights)

def average_weights(weight_list):
    """
    Federated Averaging: Returns the average of a list of weights.
    Math: w_avg = sum(w_i) / n
    """
    avg_weights = list()
    for weights_list_tuple in zip(*weight_list):
        avg_weights.append(
            np.array([np.array(w).astype(np.float32) for w in weights_list_tuple]).mean(axis=0)
        )
    return avg_weights

def train_client_local(X_train, y_train, global_weights, epochs=1):
    """
    Simulates a client training locally.
    1. Receive global model weights
    2. Train on local data
    3. Return new weights and accuracy
    """
    # Create a new local model instance
    local_model = create_resnet_model()
    
    # Initialize with global weights
    if global_weights is not None:
        set_model_weights(local_model, global_weights)
        
    # Train
    history = local_model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=8)
    
    # Return updated weights and the last accuracy
    return get_model_weights(local_model), history.history['accuracy'][-1]

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    loss, acc, recall = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc, recall
