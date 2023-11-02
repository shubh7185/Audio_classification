import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt


# Load the trained model
model = load_model('C:/Users/shubh/OneDrive/Pictures/SQL/audioclassification/audio_classification_model2.h5')
class_names = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Display the header of the web app
st.markdown(
  """<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>AUDIO CLASSIFICATION</h1><h1 style='text-align: center; color: grey;font-size:30px;margin-top:-30px;'>Using Machine Learning</h1>""",
  unsafe_allow_html=True)

# Define a function to classify the audio file
def print_prediction(file):
  """
  Classifies the audio file using the trained machine learning model.

  Args:
    file: The path to the audio file.

  Returns:
    A list of predicted classes.
  """

  audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
  mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  max_pad_len = 174  # Adjust as necessary
  pad_width = max_pad_len - mfccs.shape[1]
  mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
  prediction_feature = mfccs.reshape(1, 40, -1, 1)  # Adjust the shape according to your model's input

  predicted_vector = model.predict(prediction_feature)
  predicted_class_indices = np.argmax(predicted_vector, axis=1)
  predicted_classes = [class_names[index] for index in predicted_class_indices]

  return predicted_classes

# Define the main function of the web app
def main():
  """
  The main function of the web app.
  """

  # Display the upload file widget
  uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

  # If the user has uploaded a file, classify it and display the results
  if uploaded_file is not None:
    st.audio(uploaded_file)
    predicted_classes = print_prediction(uploaded_file)
    
     # Display the predicted class
    st.subheader("Prediction Results")
    st.info(f"Predicted Class: {predicted_classes[0]}")




    

# Run the main function
if __name__ == '__main__':
  main()
