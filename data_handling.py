# data_handling.py
from libs import tf  # Import TensorFlow from libs.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator(directory, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True, augment=False):
    """
    Create and return a data generator using ImageDataGenerator and flow_from_directory.

    Args:
    - directory: Path to the directory containing subdirectories for each class.
    - target_size: Tuple (height, width) to resize the images.
    - batch_size: Number of images per batch.
    - class_mode: 'categorical' (default) for one-hot encoding.
    - shuffle: Whether to shuffle the data (True for training, False for validation/test).
    - augment: Whether to apply data augmentation (True for training, False for validation/test).

    Returns:
    - A data generator instance.
    """
    if augment:
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rescale=1./255,         # Normalize pixel values to [0, 1]
            rotation_range=30,      # Random rotation
            width_shift_range=0.2,  # Horizontal shift
            height_shift_range=0.2, # Vertical shift
            shear_range=0.2,        # Shearing
            zoom_range=0.2,         # Random zoom
            horizontal_flip=True,   # Horizontal flip
            fill_mode='nearest'     # Fill mode for augmented pixels
        )
    else:
        # Only rescale for validation and testing
        datagen = ImageDataGenerator(rescale=1./255)

    # Create the data generator
    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,   # Resize images to the target size
        batch_size=batch_size,    # Number of images per batch
        class_mode=class_mode,    # One-hot encoding for multi-class classification
        shuffle=shuffle           # Shuffle for training; no shuffle for validation/test
    )
    return generator

def load_data(train_dir, val_dir, test_dir, target_size=(224, 224), batch_size=32):
    """
    Load and return the training, validation, and test data generators.

    Args:
    - train_dir: Path to the training dataset directory.
    - val_dir: Path to the validation dataset directory.
    - test_dir: Path to the test dataset directory.
    - target_size: Tuple (height, width) to resize the images.
    - batch_size: Number of images per batch.

    Returns:
    - A tuple of (train_generator, val_generator, test_generator).
    """
    # Training data generator with augmentation
    train_generator = create_data_generator(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,  # Shuffle training data
        augment=True   # Apply augmentation
    )

    # Validation data generator without augmentation
    val_generator = create_data_generator(
        directory=val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,  # Do not shuffle validation data
        augment=False   # No augmentation
    )

    # Test data generator without augmentation
    test_generator = create_data_generator(
        directory=test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,  # Do not shuffle test data
        augment=False   # No augmentation
    )

    return train_generator, val_generator, test_generator

# If running this script directly, test the data generators
if __name__ == "__main__":
    # Replace these paths with the actual dataset directories
    train_dir = 'faces_folder/train'
    val_dir = 'faces_folder/validation'
    test_dir = 'faces_folder/test'

    # Load data generators
    train_gen, val_gen, test_gen = load_data(train_dir, val_dir, test_dir)

    # Print class indices for verification
    print("Class mappings:", train_gen.class_indices)

    # Print some details about the datasets
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")