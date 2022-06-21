from calendar import EPOCH
from pickletools import optimize
from utilFunctions import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 0.0001
EPOCHS = 5
BATCH_SIZE = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.PATH_DATASET))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    print(imagePath)
    print(label)
    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

#baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(416,416,3)))

baseModel = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3,padding='same',
    activation= 'relu', input_shape=(256,256,3)),
    (keras.layers.MaxPooling2D(pool_size=(2,2))),
    (tf.keras.layers.Dropout(0.3)),

  (keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
    activation=tf.nn.relu)),
    (keras.layers.MaxPooling2D(pool_size = (2,2))),
    (keras.layers.Dropout(0.5)),

  (keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
    activation=tf.nn.relu)),
    (keras.layers.MaxPooling2D(pool_size = (2,2))),
    (keras.layers.Dropout(0.5)),

  (keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',
    activation=tf.nn.relu)),
    (keras.layers.MaxPooling2D(pool_size = (2,2))),
    (keras.layers.Dropout(0.5)),

  (keras.layers.Flatten()),
    (keras.layers.Dense(128,activation=tf.nn.relu)),
    (tf.keras.layers.Dropout(0.5)),
    keras.layers.Dense(64,activation='relu'),
    (tf.keras.layers.Dropout(0.25))])

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
steps_per_epoch=len(trainX)//BATCH_SIZE,
validation_data=(testX, testY),
validation_steps=len(testX)//BATCH_SIZE,
epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save(config.MODEL_PATH, save_format="h5")
print("[INFO] saving label encoder...")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
