from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import os, json
from read_params import read_params
import argparse


def generate_dataset(data_generator, raw_data_path, processed_data_path,categories, num_of_images):
  for category in categories:
    first_image = os.listdir(os.path.join(raw_data_path, category))[0]
    raw_image_path = os.path.join(os.path.join(raw_data_path, category), first_image)

    img = load_img(raw_image_path)
  #---convert the image to 3D array---
    image_data = img_to_array(img)
  #---convert into a 4-D array of 1 element of 3D array representing
  # the image---
    images_data = np.expand_dims(image_data, axis=0)

  # creating the traning image data
    generated_data = data_generator.flow(images_data, batch_size = 16,
                                      save_to_dir = os.path.join(processed_data_path, category), save_prefix='aug',save_format='jpeg')
  # saving the files
    for i in range(num_of_images):
      next(generated_data)

def make_train_dataset(config_path):
    config = read_params(config_path)

    categories_file_path = config["categories_file_path"]
    raw_train_path = config["data"]["raw"]["train"]
    processed_train_path = config["data"]["processed"]["train"]
    num_processed_train_image = config["image_parameter"]["num_processed_train_images"]
    
    category_file = json.load(open(categories_file_path))
    categories = list(category_file.keys())

    train_img_gen = ImageDataGenerator(rescale= 1.0/255, shear_range= 0.5, zoom_range= [0.5,1], 
                            horizontal_flip= True, vertical_flip = True, rotation_range=90,
                            fill_mode = 'nearest')  
      
    generate_dataset(train_img_gen, raw_train_path, processed_train_path, categories , num_processed_train_image)
    

def make_test_dataset(config_path):
    config = read_params(config_path)

    categories_file_path = config["categories_file_path"]
    raw_test_path = config["data"]["raw"]["test"]
    processed_test_path = config["data"]["processed"]["test"]
    num_processed_test_image = config["image_parameter"]["num_processed_test_images"]

   
    category_file = json.load(open(categories_file_path))
    categories = list(category_file.keys())

    
    test_img_gen = ImageDataGenerator(rescale = 1.0/255)

    generate_dataset(test_img_gen, raw_test_path, processed_test_path,categories , num_processed_test_image)

def make_directories(config_path):
   
   config = read_params(config_path)
   processed_train_path = config["data"]["processed"]["train"]
   processed_test_path = config["data"]["processed"]["test"]

   categories_file_path = config["categories_file_path"]

   category_file = json.load(open(categories_file_path))
   categories = list(category_file.keys())
   for category in categories:
      os.makedirs(os.path.join(processed_train_path, category), exit_ok = True)
      os.makedirs(os.path.join(processed_test_path, category), exist_ok=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default= "params.yaml")
    parsed_args = args.parse_args()
    config_path= parsed_args.config
