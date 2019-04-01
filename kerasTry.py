from keras import Input, Model, optimizers
from keras.applications import VGG19
from keras.applications import Xception
from keras.applications import ResNet50
from keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, AveragePooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD

train_data_dir = 'C:\\Users\\kpm\\Desktop\\CompCars\\Updated\\train\\'
test_data_dir = 'C:\\Users\\kpm\\Desktop\\CompCars\\Updated\\test\\'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 75
batch_size = 32

img_width, img_height = 224, 224

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1. / 255,
    validation_split=0.2, )  # set validation split

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training')  # set as training data

validation_generator = validation_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    save_to_dir="C:\\Users\\kpm\\Desktop\\CompCars\\Updated\\valid")  # set as validation data

test_generator = train_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42,
    save_to_dir="C:\\Users\\kpm\\Desktop\\CompCars\\Updated\\SmallSet\\test"
)

base_model = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(12, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=30,
                              validation_data=validation_generator,
                              epochs=50,
                              verbose=1,
                              workers=1,
                              use_multiprocessing=False,
                              validation_steps=3090 // 32)
model.save('model.h5')
model.save_weights('model_weights.h5')
model.save_weights('model_weights_json.json')