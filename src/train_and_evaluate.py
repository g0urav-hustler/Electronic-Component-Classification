from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D

import argparse

from tflite_model_maker import ImageClassifierDataLoader
data = ImageClassifierDataLoader.from_folder('aug_images')

# importing mobilenet model 
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import Xception

def MobileNet_model(config_path):
    # initializing mobilenet model
  mobilenet = MobileNet(input_shape= image_size + [3] , weights = 'imagenet', include_top= False)

  # we don't have to train the existing weights
  for layer in mobilenet.layers:
    layer.trainable = False

  # globalaveragepooling2d layer
  mblnt_gap2d_layer = GlobalAveragePooling2D()(mobilenet.output)

  # making dense layer
  mblnt_dense_layer = Dense(units = len(categories), activation= 'softmax')(mblnt_gap2d_layer)

  # model object
  mobilenet_model = Model(inputs = mobilenet.input, outputs = mblnt_dense_layer )

  # compiling the model
  mobilenet_model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
  # fitting the model
  mobilenet_result = mobilenet_model.fit_generator(train_data, validation_data= test_data, epochs = 30)

  model_folder = "/content/drive/MyDrive/ML Projects/Electronic Component Classification"
  model_name = 'mobilenet_model_82acc_30ep.h5'
  model_path = model_folder + '/' + model_name

  # saving the model
  mobilenet_model.save(model_path)


### inceptionV3 Model
# importing inceptionv3
def inception_model(config)
  
  # initializing inception model
  inception = InceptionV3(input_shape= image_size + [3] , weights = 'imagenet', include_top= False)
  # we don't have to train the existing weights
  for layer in inception.layers:
    layer.trainable = False
  # # flattening the inception layer
  # inv3_flat_layer = Flatten()(inception.output)

  # globalaveragepooling2d layer
  inv3_gap2d_layer = GlobalAveragePooling2D()(inception.output)

  # final dense layer
  inv3_dense_layer = Dense(units = len(categories), activation= 'softmax')(inv3_gap2d_layer)

  # model object
  inceptionv3_model = Model(inputs = inception.input, outputs = inv3_dense_layer )

  # compiling the model
  inceptionv3_model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

  # fitting the model
  inceptionv3_result = inceptionv3_model.fit_generator(train_data, validation_data= test_data, epochs = 40)

  model_folder = "/content/drive/MyDrive/ML Projects/Electronic Component Classification"
  inceptionv3_model_name = 'inceptionv3_model_92vacc_100tacc_40ep.h5'
  inceptionv3_model_path = model_folder + '/' + inceptionv3_model_name
  # saving the inception3 model
  inceptionv3_model.save(inceptionv3_model_path)


# Xception Model
def Xception_model(config_path):

  # initializing the xception model
  xception = Xception(input_shape = image_size + [3], weights = 'imagenet', include_top= False)
  # not train the layers
  for layer in xception.layers:
    layer.trainable = False

  # # flat layer
  # xcpt_flat_layer = Flatten()(xception.output)

  # globalaveragpooling layer
  xcept_gap2d_layer = GlobalAveragePooling2D()(xception.output)

  # dense layer
  xcept_dense_layer = Dense(units = len(categories), activation= 'softmax')(xcept_gap2d_layer)

  # building model
  xception_model = Model(inputs = xception.input , outputs = xcept_dense_layer)

  # compiling model
  xception_model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

  # xception model summary
  xception_model.summary()

  # fitting xception model
  xception_result = xception_model.fit_generator(train_data, validation_data= test_data, epochs = 30)

  model_folder = "/content/drive/MyDrive/ML Projects/Electronic Component Classification"
  xception_model_name = 'xception_model_92vacc_98tacc_30ep.h5'
  xception_model_path = model_folder + '/' + xception_model_name

  xception.save(xception_model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default= "params.yaml")
    parsed_args = args.parse_args()