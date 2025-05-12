import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# ✅ 1. Set dataset path
# Option 1: Use raw string
DATASET_PATH = r"C:\Users\Dayakar2431\OneDrive\Desktop\TejaFile\Rice_Image_Dataset\Rice_Image_Dataset"


if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found at {DATASET_PATH}")

# ✅ 2. Load Images & Labels
images, labels = [], []
for subfolder in os.listdir(DATASET_PATH):
    subfolder_path = os.path.join(DATASET_PATH, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    for image_filename in os.listdir(subfolder_path):
        images.append(os.path.join(subfolder_path, image_filename))
        labels.append(subfolder)

df = pd.DataFrame({'image': images, 'label': labels})

# ✅ 3. Convert images to arrays and resize to (128x128)
image_size = (128, 128)
X, y = [], []

for i in range(len(df)):
    img = load_img(df['image'][i], target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize (0-1)
    X.append(img_array)
    y.append(df['label'][i])

# ✅ 4. Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# ✅ 5. Encode labels using one-hot encoding
label_encoder = LabelBinarizer()
y = label_encoder.fit_transform(y)

# ✅ 6. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 7. Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# ✅ 8. Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# ✅ 9. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 10. Callbacks (Early Stopping & Model Checkpoint)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("model/seed_classifier.h5", save_best_only=True)
]

# ✅ 11. Train the Model
history = model.fit(train_generator, epochs=15, validation_data=(X_test, y_test), callbacks=callbacks)

# ✅ 12. Save Label Encoder
np.save("model/label_classes.npy", label_encoder.classes_)

# ✅ 13. Plot Training Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.savefig("static/results/training_performance.png")
plt.show()

print("✅ Model Training Completed & Saved!")
