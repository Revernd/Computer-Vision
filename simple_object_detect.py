import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import matplotlib.pyplot as plt
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Creating Datasets
(X_num, y_num), _ = keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0

grid_size = 16  # image_size / mask_size

def make_numbers(X, y):
    for _ in range(3):
        # pickup random index
        idx = np.random.randint(len(X_num))

        # make digits colorful
        number = X_num[idx] @ (np.random.rand(1, 3) + 0.1)
        number[number > 0.1] = np.clip(number[number > 0.1], 0.5, 0.8)

        # class of digit
        cls = y_num[idx]

        # random position for digit
        px, py = np.random.randint(0, 100), np.random.randint(0, 100)

        # digit belong to which mask position
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size
        channels = y[my][mx]

        # prevent duplicates
        if channels[0] > 0:
            continue

        channels[0] = 1.0
        channels[1] = px - (mx * grid_size)  # x1
        channels[2] = py - (my * grid_size)  # y1
        channels[3] = 28.0  # x2, in this demo image only 28 px as width
        channels[4] = 28.0  # y2, in this demo image only 28 px as height
        channels[5 + cls] = 1.0

        # put digit in X
        X[py:py + 28, px:px + 28] += number

def make_data(size=64):
    X = np.zeros((size, 128, 128, 3), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)
    for i in range(size):
        make_numbers(X[i], y[i])

    X = np.clip(X, 0.0, 1.0)
    return X, y

def get_color_by_probability(p):
    if p < 0.3:
        return (1., 0., 0.)
    if p < 0.7:
        return (1., 1., 0.)
    return (0., 1., 0.)

def show_predict(X, y, threshold=0.1):
    X = X.copy()  # Make a copy of the image to draw on
    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, x2, y2 = channels[:5]

            # if prob < threshold we won't show anything
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)
            # bounding box
            px, py = (mx * grid_size) + x1, (my * grid_size) + y1
            cv2.rectangle(X, (int(px), int(py)), (int(px + x2), int(py + y2)), color, 1)

            # label
            cv2.rectangle(X, (int(px), int(py - 10)), (int(px + 12), int(py)), color, -1)
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))

    plt.imshow(X)
    plt.axis('off')  # Hide the axes for better visual
    plt.show()  # Display the image

class SimpleObjectDetectModel(tf.keras.Model):
    def __init__(self):
        super(SimpleObjectDetectModel, self).__init__()

        self.conv1 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D()
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D()
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool4 = layers.MaxPool2D()
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool5 = layers.MaxPool2D()
        self.bn5 = layers.BatchNormalization()
        

        self.prob = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='x_prob')
        self.boxes = layers.Conv2D(4, kernel_size=3, padding='same', name='x_boxes')
        self.cls = layers.Conv2D(10, kernel_size=3, padding='same', activation='sigmoid', name='x_cls')

    def call(self, x):
        x = self.bn1(self.pool1(self.conv1(x)))
        x = self.bn2(self.conv2(x))       
        x = self.bn3(self.pool3(self.conv3(x))) 
        x = self.bn4(self.pool4(self.conv4(x)))
        x = self.bn5(self.pool5(self.conv5(x)))

        x_prob = self.prob(x)
        x_boxes = self.boxes(x)
        x_cls = self.cls(x)

        gate = tf.where(x_prob > 0.5, tf.ones_like(x_prob), tf.zeros_like(x_prob))
        x_boxes = x_boxes * gate
        x_cls = x_cls * gate

        x = tf.concat([x_prob, x_boxes, x_cls], axis=-1)
        return x

# Instantiate and build the model
model = SimpleObjectDetectModel()
model.build(input_shape=(None, 128, 128, 3))
print(model.summary())

idx_p = [0]
idx_bb = [1, 2, 3, 4]
idx_cls = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Define custom losses dictionary
losses = {
    "prob": tf.keras.losses.BinaryCrossentropy(),
    "boxes": tf.keras.losses.MeanSquaredError(),
    "cls": tf.keras.losses.BinaryCrossentropy()
}

# Define the loss function that combines the custom losses
def combined_loss(y_true, y_pred):
    # Extract individual components from the prediction and ground truth
    y_true_prob = tf.gather(y_true, idx_p, axis=-1)
    y_pred_prob = tf.gather(y_pred, idx_p, axis=-1)
    y_true_boxes = tf.gather(y_true, idx_bb, axis=-1)
    y_pred_boxes = tf.gather(y_pred, idx_bb, axis=-1)
    y_true_cls = tf.gather(y_true, idx_cls, axis=-1)
    y_pred_cls = tf.gather(y_pred, idx_cls, axis=-1)

    # Compute the loss for each component
    loss_prob = losses["prob"](y_true_prob, y_pred_prob)
    loss_boxes = losses["boxes"](y_true_boxes, y_pred_boxes)
    loss_cls = losses["cls"](y_true_cls, y_pred_cls)

    # Combine the losses
    return tf.reduce_mean(loss_prob + loss_boxes + loss_cls)

# Compile the model with the combined loss function
opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss=combined_loss, optimizer=opt, metrics=['accuracy'])

def preview(numbers=None, threshold=0.1):
    X, y = make_data(size=1)
    y = model.predict(X)
    show_predict(X[0], y[0], threshold=threshold)

# Call the preview function to test
preview(threshold=0.7)

batch_size = 32
X_train, y_train = make_data(size=batch_size * 250)

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, shuffle=True, verbose=True)
plt.plot(history.history['loss'])
model.save('./saved_models/odj_detector.h5')
preview(threshold=0.7)
