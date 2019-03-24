from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from deneme.pyimagesearch.smallervggnet import SmallerVGGNet
from keras.optimizers import Adam

train_data_dir = '/run/media/toorn/New Volume/SonData/train/'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 75
batch_size = 32

img_width, img_height = 100, 100

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,)  # set validation split

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training', color_mode="grayscale")  # set as training data

validation_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation', color_mode="grayscale")  # set as validation data

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(12))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=img_width,
                            height=img_height,
                            depth=1, finalAct="sigmoid")

INIT_LR = 1e-3
EPOCHS = 75
BS = 32

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=1000 // BS,
                        epochs=EPOCHS, verbose=1,
                        validation_steps=200 // BS)

model.save_weights('first_try.h5')
model.save('first_model.h5')