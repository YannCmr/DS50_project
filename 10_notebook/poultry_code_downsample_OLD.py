import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder, target_size=(224, 224), color_mode='YCbCr', chroma_downsample=2):
    images = []
    labels = []

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                if img_path.endswith('.JPG'):
                    img = Image.open(img_path)

                    if color_mode == 'RGB':
                        img = img.convert('RGB')
                        img = img.resize(target_size)
                        img = np.array(img) / 255.0
                    elif color_mode == 'YCbCr':
                        img = img.convert('YCbCr')
                        img = img.resize(target_size)
                        ycbcr = np.array(img).astype(np.float32) / 255.0
                        Y = ycbcr[:, :, 0:1]

                        Cb = ycbcr[:, :, 1]
                        Cr = ycbcr[:, :, 2]

                        if chroma_downsample > 1:
                            Cb_sub = Cb[::chroma_downsample, ::chroma_downsample]
                            Cr_sub = Cr[::chroma_downsample, ::chroma_downsample]

                            Cb = np.kron(Cb_sub, np.ones((chroma_downsample, chroma_downsample)))
                            Cr = np.kron(Cr_sub, np.ones((chroma_downsample, chroma_downsample)))
                            Cb = Cb[:target_size[0], :target_size[1]]
                            Cr = Cr[:target_size[0], :target_size[1]]

                        Cb = Cb[..., np.newaxis]
                        Cr = Cr[..., np.newaxis]

                        img = np.concatenate((Y, Cb, Cr), axis=2)
                    else:
                        raise ValueError("color_mode must be 'RGB' or 'YCbCr'")

                    images.append(img)
                    labels.append(label)

    return np.array(images), np.array(labels)

def prepare_data(train_folder, test_folder, target_size=(224, 224), chroma_downsample=2):
    X_train, y_train = load_images_from_folder(train_folder, target_size, color_mode='RGB', chroma_downsample=chroma_downsample)
    X_test, y_test = load_images_from_folder(test_folder, target_size, color_mode='RGB', chroma_downsample=chroma_downsample)

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)
    y_test_int = label_encoder.transform(y_test)

    y_train_cat = to_categorical(y_train_int)
    y_test_cat = to_categorical(y_test_int)

    return (X_train, y_train_cat), (X_test, y_test_cat), label_encoder


DATA_FOLDER_PATH = "../"
OUTPUT_DIR = DATA_FOLDER_PATH + "00_archive/data_samples/"
train_folder = OUTPUT_DIR + "train"
test_folder = OUTPUT_DIR + "test"
CHROMA_DOWNSAMPLE = 4  

(X_train, y_train), (X_test, y_test), label_encoder = prepare_data(train_folder, test_folder, chroma_downsample=CHROMA_DOWNSAMPLE)

print(f'Training data shape: {X_train.shape}')
print(f'Test data shape: {X_test.shape}')

plt.imshow(X_train[0])
plt.title("YCbCr image with chroma downsampling")
plt.show()

unique_labels, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
unique_labels = unique_labels.astype(int)
label_names = label_encoder.inverse_transform(unique_labels)

plt.figure(figsize=(10, 6))
sns.barplot(x=label_names, y=counts, hue=label_names, palette='viridis', dodge=False)
plt.title('Distribution of Labels in Training Data')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_names)), labels=[''] * len(label_names))
plt.legend(title='Labels', loc='upper right')
plt.show()

model = Sequential()
model.add(Input(shape=(224, 224, 3)))

model.add(Conv2D(32, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

metrics = pd.DataFrame(model.history.history)
metrics[['loss', 'val_loss']].plot()
metrics[['accuracy', 'val_accuracy']].plot()

model.evaluate(X_test, y_test)
predictions = np.argmax(model.predict(X_test), axis=1)
print(classification_report(np.argmax(y_test, axis=1), predictions, target_names=label_encoder.classes_))

sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), predictions), annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.show()
