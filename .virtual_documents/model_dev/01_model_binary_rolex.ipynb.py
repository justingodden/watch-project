import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np.random.seed(0)


df_orig = pd.read_csv('../wf_df_raw.csv')


df = df_orig.copy()


BASE_DIR = os.path.dirname(os.getcwd())

RAW_IMGS_DIR = os.path.join(BASE_DIR, 'raw_images')

IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
TRAINING_DIR = os.path.join(IMAGES_DIR, 'Training')
VALIDATION_DIR = os.path.join(IMAGES_DIR, 'Validation')

TRAIN_ROLEX_DIR = os.path.join(TRAINING_DIR, 'Rolex')
TRAIN_NOT_ROLEX_DIR = os.path.join(TRAINING_DIR, 'Not_Rolex')
VAL_ROLEX_DIR = os.path.join(VALIDATION_DIR, 'Rolex')
VAL_NOT_ROLEX_DIR = os.path.join(VALIDATION_DIR, 'Not_Rolex')


try:
    shutil.rmtree(IMAGES_DIR)
except: pass

os.mkdir(IMAGES_DIR)
os.mkdir(TRAINING_DIR)
os.mkdir(VALIDATION_DIR)

os.mkdir(TRAIN_ROLEX_DIR)
os.mkdir(TRAIN_NOT_ROLEX_DIR)
os.mkdir(VAL_ROLEX_DIR)
os.mkdir(VAL_NOT_ROLEX_DIR)


df = df.dropna(subset=['img_name'])


for index, row in df.iterrows():
    
    CURR_IMG_DIR = os.path.join(RAW_IMGS_DIR, row['img_name'])
    
    random_number = np.random.randint(1,101)
    
    if random_number <= 75 and row['brand'] == 'Rolex':
        shutil.copy(CURR_IMG_DIR, TRAIN_ROLEX_DIR)
    
    elif random_number <= 75 and row['brand'] get_ipython().getoutput("= 'Rolex':")
        shutil.copy(CURR_IMG_DIR, TRAIN_NOT_ROLEX_DIR)
    
    elif random_number > 75 and row['brand'] == 'Rolex':
        shutil.copy(CURR_IMG_DIR, VAL_ROLEX_DIR)
    
    elif random_number > 75 and row['brand'] get_ipython().getoutput("= 'Rolex':")
        shutil.copy(CURR_IMG_DIR, VAL_NOT_ROLEX_DIR)


print(f"Train Rolex num: {len(os.listdir(TRAIN_ROLEX_DIR))}")
print(f"Train Not Rolex num: {len(os.listdir(TRAIN_NOT_ROLEX_DIR))}")
print(f"Val Rolex num: {len(os.listdir(VAL_ROLEX_DIR))}")
print(f"Val Not Rolex num: {len(os.listdir(VAL_NOT_ROLEX_DIR))}")
print()
print(round(100 * (1 - len(os.listdir(TRAIN_ROLEX_DIR)) / len(os.listdir(TRAIN_NOT_ROLEX_DIR))), 2))
print(round(100 * (1 - len(os.listdir(VAL_ROLEX_DIR)) / len(os.listdir(VAL_NOT_ROLEX_DIR))), 2))


train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(300, 300),
    class_mode='binary'
)


validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(300, 300),
    class_mode='binary'
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)


from PIL import Image

folder_path = TRAINING_DIR
extensions = ['.jpg']
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        im = Image.open(file_path)
        rgb_im = im.convert('RGB')
        if filee.split('.')[1] not in extensions:
            extensions.append(filee.split('.')[1])
