from libs import tf
import matplotlib.pyplot as plt
import pickle
def load_history(historypath):
    with open(historypath, 'rb')  as f:
        history = pickle.load(f)
    
    return history

# Save the model
def create_graph(history):
    

    # Plot training and validation metrics
    accuracy = history.model["accuracy"]
    val_accuracy = history.model["val_accuracy"]
    loss = history.model["loss"]
    val_loss = history.model["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, "bo-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "ro-", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    historypath = 'training_history.pkl'
    history = load_history

    
    create_graph(history)