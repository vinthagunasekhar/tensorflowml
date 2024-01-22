import urllib.request
import zipfile

url="https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
validation_url="https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
file_name="horse-or-human.zip"
validation_file_name="validation-horse-or-human.zip"
validation_dir='horse-or-huma/validation/'
training_dir='horse-or-human/training/'
urllib.request.urlretrieve(url,file_name)
urllib.request.urlretrieve(validation_url,validation_file_name)
zip_ref= zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()
zip_ref= zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers.legacy import RMSprop

train_datagen= ImageDataGenerator(rescale=1/255)

train_generator= train_datagen.flow_from_directory(
    training_dir,
    target_size=(300,300),
    class_mode='binary'
)

validation_datagen= ImageDataGenerator(rescale=1/255)
validation_generator= train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300),
    class_mode='binary'
)

weights_url="https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file="inception_v3.h5"
urllib.request.urlretrieve(weights_url,weights_file)

pre_trained_model= InceptionV3(input_shape=(150, 150, 3), include_top=False, weights= None)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable= False

last_layer= pre_trained_model.get_layer('mixed7')
last_output= last_layer.output

x=layers.Flatten()(last_output)
x=layers.Dense(1024, activation='relu')(x)
x=layers.Dense(1, activation='sigmoid')(x)
model= Model(pre_trained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics=['acc'])
history=model.fit_generator(train_generator, epochs=15)

history=model.fit_generator(train_generator, epochs=15, validation_data=validation_generator)