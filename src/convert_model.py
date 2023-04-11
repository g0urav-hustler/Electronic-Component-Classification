# importing tensorflow
import tensorflow as tf

def convert_model(config_path):

  # converter object
  converter = tf.lite.TFLiteConverter.from_keras_model(xception_model)

  # converting the model
  tflite_model = converter.convert()
  xception_tf_model = model_folder + "/" + "ecs_model.tflite"

  # saving the model
  with open(xception_tf_model, 'wb') as f:
    f.write(tflite_model)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default= "params.yaml")
    parsed_args = args.parse_args()
    convert_model(parsed_args.config)