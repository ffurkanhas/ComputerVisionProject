from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import pandas as pd

test_data_dir = '/run/media/toorn/New Volume/SonData/test/'
train_data_dir = '/run/media/toorn/New Volume/SonData/train/'

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(100, 100),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,)  # set validation split

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical',
    subset='training', color_mode="grayscale")  # set as training data

model = load_model('first_model.h5')
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)

print(pred)
predicted_class_indices = np.argmax(pred, axis=1)
print(predicted_class_indices)
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})
results.to_csv("results.csv", index=False)