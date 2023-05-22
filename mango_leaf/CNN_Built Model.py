#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
data_dir = pathlib.Path('DATASET') #Path of Photos
image_count = len(list(data_dir.glob('*/*.jpeg')))
print(image_count)
# rocketlaunches = list(data_dir.glob('rocketlaunches/*'))
# im = Image.open(str(rocketlaunches[0]))
batch_size = 32
img_height = 100
img_width = 100
data_train = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
print("Your Training Data : ",  data_train)
data_test = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2, subset="validation",seed=123, image_size=(img_height, img_width),batch_size=batch_size)
class_names = data_train.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in data_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
print(model.summary())
epochs=10
Image_Model = model.fit(data_train,validation_data=data_test,epochs=epochs)
print('Accuracy Status : ',Image_Model.history['accuracy'])

#testing the model

img = tf.keras.utils.load_img('check.jpg', target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
result = model.predict(img_array)
print(result)
print(f"This is image of {class_names[np.argmax(result)]}")



#confusion matrix graph

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate predictions on the test set
y_true = np.concatenate([y for x, y in data_test], axis=0)
y_scores = model.predict(data_test)
y_pred = np.argmax(y_scores, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap='Blues')

# Add title and axis labels
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

# Add class labels to the x and y axis ticks
tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Set alignment and color of the label text
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor", color="white")
plt.setp(ax.get_yticklabels(), color="white")

# Loop over data dimensions and create text annotations
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Show the plot
plt.tight_layout()
plt.show()




#F1 score graph


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Generate predictions on the test set
y_true = np.concatenate([y for x, y in data_test], axis=0)
y_scores = model.predict(data_test)
y_pred = np.argmax(y_scores, axis=1)

# Compute the F1 score for each class
f1_scores = f1_score(y_true, y_pred, average=None)

# Plot the F1 score graph
fig, ax = plt.subplots()
ax.bar(class_names, f1_scores)
ax.set_xlabel('Class')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score per Class')

# Add the F1 score values as text annotations on top of each bar
for i, f1_score in enumerate(f1_scores):
    ax.text(i, f1_score, f'{f1_score:.2f}', ha='center', va='bottom')

plt.show()




#Precision Graph


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# Generate predictions on the test set
y_true = np.concatenate([y for x, y in data_test], axis=0)
y_scores = model.predict(data_test)
y_pred = np.argmax(y_scores, axis=1)

# Compute the precision for each class
precision_scores = precision_score(y_true, y_pred, average=None)

# Plot the precision graph
fig, ax = plt.subplots()
ax.bar(class_names, precision_scores)
ax.set_xlabel('Class')
ax.set_ylabel('Precision')
ax.set_title('Precision per Class')

# Add the precision values as text annotations on top of each bar
for i, precision_score in enumerate(precision_scores):
    ax.text(i, precision_score, f'{precision_score:.2f}', ha='center', va='bottom')

plt.show()


# %%
