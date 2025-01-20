from libs import *

def model_build():
    # --- Build the Model ---
    # Load the pretrained MobileNetV2 model without the top layers
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Freeze the base model to avoid training it
    base_model.trainable = False

    # Build the model with custom layers
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # For 3 classes: Denzel, Imaan, Sandra
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model #returns model

 


if __name__ == "__main__":
    #build model
    model = model_build()
    model.summary() #print model for verification




