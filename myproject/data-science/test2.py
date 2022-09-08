import tensorflow as tf
from tensorflow import keras
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import keras_tuner as kt


def get_sample(labels,images,n):
    train_images=[]
    train_labels=[]
    for i in range(10):
        idx=np.random.choice(np.where(labels==i)[0],int(n/10),replace=False)
        idx.sort()
        train_images.append(np.array(images)[idx])
        train_labels.append(np.array(labels)[idx])
    train_images=np.reshape(train_images,(n,32,32,3))    
    train_labels=np.reshape(train_labels,(n))

    return train_labels,train_images



(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_labels,train_images,=get_sample(train_labels,train_images, 10000)
test_labels,test_images=get_sample(test_labels,test_images,2000)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def model_builder():               
    model = tf.keras.models.Sequential()
    
    # layers to resize the images to a consistent shape and to rescale pixel values
    IMG_SIZE=32
    #model.add(tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE))
    model.add(tf.keras.layers.Rescaling(1./255))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model                


model=model_builder()
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy :', accuracy)                   