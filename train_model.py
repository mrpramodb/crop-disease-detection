import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from dataset_loader import train_generator

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),  
    MaxPooling2D(pool_size=(2,2)),  
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),  
    Dense(128, activation='relu'),  
    Dropout(0.5),  
    Dense(len(train_generator.class_names), activation='softmax')  
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model using the generator and ignore incomplete batches
model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))

# Save the trained model
model.save("crop_disease_model.h5")

print("✅ Model trained and saved successfully!")
