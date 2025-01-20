# training.py
from model_arc import model_build
from data_handling import load_data

import pickle

def train_model():
    """
    Train the model with training and validation datasets and evaluate it on the test dataset.
    """
    # Dataset directories
    train_dir = 'faces_folder/train'
    val_dir = 'faces_folder/validation'
    test_dir = 'faces_folder/test'

    # Load data
    train_generator, val_generator, test_generator = load_data(train_dir, val_dir, test_dir)

    # Print class mappings for debugging
    print("Class mappings:", train_generator.class_indices)

    # Build the model
    model = model_build()

    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Save the trained model
    model.save('trained_model.h5')
    print("Model saved to 'trained_model.h5'")

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("history saved")

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    return history

if __name__ == "__main__":
    train_model()
