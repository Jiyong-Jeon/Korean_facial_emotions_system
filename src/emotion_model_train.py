import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

batch_size = 32
img_size = 224
epochs = 100

# 데이터 수 출력
data_dir = './datasets'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob("blank/*.jpg")))
print(image_count)
image_count = len(list(data_dir.glob("funny/*.jpg")))
print(image_count)
image_count = len(list(data_dir.glob("sad/*.jpg")))
print(image_count)
image_count = len(list(data_dir.glob("angry/*.jpg")))
print(image_count)

CLASS_NAMES = ['angry', 'blank', 'funny', 'sad']

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.2)

train_ds = train_datagen.flow_from_directory(
    data_dir,
    batch_size=batch_size,
    shuffle=True,
    target_size=(img_size, img_size),
    classes = list(CLASS_NAMES),
    subset='training')

val_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation')

base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                               include_top=False, input_tensor=None,
                                               weights='imagenet')

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

initializer = tf.keras.initializers.HeNormal()
predict_layer = tf.keras.models.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax')
])

inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs)
outputs = predict_layer(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

# model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), 
#               loss='categorical_crossentropy',
#               metrics=['acc'])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True), 
              loss='categorical_crossentropy',
              metrics=['acc'])
 
# 모델 저장 조건 설정
modelpath = './models/emotion/test{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer= tf.keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
# 학습 자동 중단 설정
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(train_ds, epochs=epochs,  batch_size=batch_size, validation_data=val_ds, callbacks=[checkpointer, early_stopping_callback])

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()