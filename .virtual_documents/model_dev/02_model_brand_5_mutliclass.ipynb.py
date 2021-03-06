import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import random

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import cv2


np.random.seed(0)


df_orig = pd.read_csv('../wf_df_raw.csv')


df = df_orig.copy()


df = df.dropna(subset=['img_name'])


df['brand'] = df['brand'].map(lambda x: x.replace(' ', '_'))


brand_list = list(df['brand'].value_counts()[:5].index)


brand_list.sort()


BASE_DIR = os.path.dirname(os.getcwd())

RAW_IMGS_DIR = os.path.join(BASE_DIR, 'raw_images')

IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
TRAINING_DIR = os.path.join(IMAGES_DIR, 'Training')
VALIDATION_DIR = os.path.join(IMAGES_DIR, 'Validation')


try:
    shutil.rmtree(IMAGES_DIR)
except: pass

os.mkdir(IMAGES_DIR)
os.mkdir(TRAINING_DIR)
os.mkdir(VALIDATION_DIR)


for brand in brand_list:
    os.mkdir(os.path.join(TRAINING_DIR, brand))
    os.mkdir(os.path.join(VALIDATION_DIR, brand))


for index, row in df.iterrows():
    
    CURR_IMG_DIR = os.path.join(RAW_IMGS_DIR, row['img_name'])
    
    random_number = np.random.randint(1,101)
    
    if row['brand'] in brand_list and random_number <= 80:
        shutil.copy(CURR_IMG_DIR, os.path.join(TRAINING_DIR, row['brand']))
        
    elif row['brand'] in brand_list and random_number > 80:
        shutil.copy(CURR_IMG_DIR, os.path.join(VALIDATION_DIR, row['brand']))


train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)


validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
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
    tf.keras.layers.Dense(5, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15
)


pre_trained_model = tf.keras.applications.InceptionV3(
    include_top = False,
    weights = "imagenet",
    input_shape = (300, 300, 3)
)


for layer in pre_trained_model.layers:
    layer.trainable = False


last_layer = pre_trained_model.layers[-1]
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a final softmax layer for classification
x = tf.keras.layers.Dense(5, activation='softmax')(x)


model = tf.keras.models.Model(pre_trained_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    batch_size=32,
    epochs=45
)


model.evaluate(validation_generator)


BRAND = random.choice(brand_list)
BRAND_DIR = os.path.join(VALIDATION_DIR, BRAND)
WATCHES = os.listdir(BRAND_DIR)
WATCH = random.choice(WATCHES)
WATCH_DIR = os.path.join(BRAND_DIR, WATCH)


img = image.load_img(WATCH_DIR, target_size=(300, 300))
x = image.img_to_array(img)
x = x / 255.
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)


img_orig = cv2.imread(WATCH_DIR)[...,::-1]
imgplot = plt.imshow(img_orig)
print(f"Actual:\t\t{BRAND}\nPredicted:\t{brand_list[np.argmax(prediction)]}")


correct = 0
n = 200

for i in range(n):
    BRAND = random.choice(brand_list)
    BRAND_DIR = os.path.join(VALIDATION_DIR, BRAND)
    WATCHES = os.listdir(BRAND_DIR)
    WATCH = random.choice(WATCHES)
    WATCH_DIR = os.path.join(BRAND_DIR, WATCH)
    
    img = image.load_img(WATCH_DIR, target_size=(300, 300))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    
    if BRAND == brand_list[np.argmax(prediction)]:
        correct += 1

print(round(100 * correct/n, 2))
