import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_resnet_model(input_shape=(75, 75, 3)):
    """
    Creates a ResNet50V2-based model for Lung Cancer Detection.
    Uses Transfer Learning with ImageNet weights.
    """
    # Base model with pre-trained weights
    base_model = applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers to retain learned features
    base_model.trainable = False 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Regularization
        layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    # Use a lower learning rate for fine-tuning/transfer learning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    return model

