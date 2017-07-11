import os
import cv2
import keras
import time as t
import numpy as np
import pandas as pd
import random as rm
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout, Flatten, Dense

t1 = t.time()
model = VGG19(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
# for layer in model.layers[:17]:
#     layer.trainable = False
#     print layer
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)
model_final = Model(inputs = model.input, outputs = predictions)
model_final.compile(loss = "categorical_crossentropy",
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])
# print model_final.summary()
folder_name = '01/'
def get_frames(start_path = folder_name):
    for filenames in os.walk(start_path):
        filenames
    return filenames[2]
total_Frames = get_frames()
data = np.zeros((224, 224, 3, np.size(total_Frames)))
for frame in total_Frames:
    data[:, :, :, int(frame.split('.')[0])-1] = cv2.resize(cv2.imread(folder_name+frame),
                                                           (224, 224)).astype(np.float32)
labels = np.asarray(pd.read_csv('01.csv')['Label'])

dataSize = 300
nums = [x for x in range(dataSize)]
rm.shuffle(nums)
r1 = int (np.multiply(0.80, dataSize))
r2 = dataSize - r1
s1 = nums[0:r1]
s2 = nums[r1:dataSize]
train = data[:, :, :, s1[:]]
test = data[:, :, :, s2[:]]
ztrain = np.zeros((r1, 224, 224, 3))
ztest = np.zeros((r2, 224, 224, 3))

for i in range(r1):
    ztrain[i, :, :, :] = train[:, :, :, i]
for j in range(r2):
    ztest[j, :, :, :] = test[:, :, :, j]

trainLabel, testLabel = labels[s1[:]], labels[s2[:]]
num_classes = 2
y_train = keras.utils.to_categorical(trainLabel, num_classes)
y_test = keras.utils.to_categorical(testLabel, num_classes)
x_train = ztrain.astype('float32')
x_test = ztest.astype('float32')

print "Data Dimensions: ",(x_train.shape, y_train.shape), (x_test.shape, y_test.shape)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
hist = model_final.fit(x_train, y_train, batch_size=50, epochs=100, verbose=2,
                 callbacks=[early], validation_split=0.20, shuffle=True)
score = model_final.evaluate(x_test, y_test, verbose=2)
Val_acc, train_Acc = hist.history['val_acc'], hist.history['acc']
Avg_train_acc, Avg_val_Acc = np.average(train_Acc), np.average(Val_acc)
print('Train Accuracy:      \t', Avg_train_acc *100)
print('Validation Accuracy: \t', Avg_val_Acc *100)
print('Test accuracy:       \t', score[1]*100)
print('Test loss:           \t', score[0])
t2 = t.time()
print "Total Time: ", (t2-t1)
# plot_model(model_final, to_file='VIPcup.png',show_shapes=True)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
