from tflite_runtime.interpreter import Interpreter
import streamlit as st
from PIL import Image
import numpy as np

model_path = "models/quantized_model.tflite"


component_dict = {0: 'Ammature', 1: 'Automative fuse', 2: 'Battery', 3: 'Cartridge Fuse', 
                    4: 'Ceramic Capacitor', 5: 'Clip Lead', 6: 'Crystals', 7: 'Diode',
                    8: 'Electrolytic Capcitor', 9: 'Heat Sink', 10: 'Inductor', 11: 'Led',
                    12: 'Microprocessor', 13: 'Mosfet', 14: 'Motor', 15: 'Potentiometer',
                    16: 'Resistor', 17: 'Rheostat', 18: 'Shunt', 19: 'Switch',
                    20: 'Transformer', 21: 'Transitor'}

#title
st.title("Electronic Component Scanner")

#subtitle
st.markdown("This application helps you to classify component by their images.")

@st.cache_resource(show_spinner="Loading the app..")
def load_model(model_name):
    interpreter = Interpreter(model_path = model_name)

    return interpreter


interpreter = load_model(model_path)
input_details = interpreter.get_input_details()
  
  
input_index = input_details[0]["index"]
image_shape = input_details[0]['shape'][1:-1]


#image uploader
image = st.file_uploader(label = "Upload your image here", type=['png','jpg','jpeg'])

if image is not None:

    img = Image.open(image) 
    
    with st.spinner("Classifing the image..."):
        img = img.resize(image_shape)
        st.image(img)
        input_tensor = np.array(np.expand_dims(img,0))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        pred_output_details = interpreter.get_output_details()

        output_data = interpreter.get_tensor(pred_output_details[0]['index'])
        pred = np.squeeze(output_data)
        highest_pred_loc = np.argmax(pred)

        component_name = component_dict[highest_pred_loc]

        
        st.write("This is a ", component_name)
else:
    st.write("Upload an Image first")


st.caption("Made by Gourav Chouhan ")
