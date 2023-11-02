import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load your trained model
model = load_model(r'C:\Users\shubh\OneDrive\Pictures\SQL\audioclassification\audio_classification_model2.h5')

max_pad_len = 174  # Set the appropriate value for max_pad_len
num_rows = 40
num_columns = 174
num_channels = 1





# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 
# Load other necessary variables such as le, num_rows, num_columns, num_channels

# Define the prediction function
def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict(prediction_feature)
    predicted_class_indices = np.argmax(predicted_vector, axis=1)
    predicted_class = le.inverse_transform(predicted_class_indices) 
    st.write("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba = predicted_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        st.write(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        
def extract_features(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')  # Load audio without resampling
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]  # Corrected variable name
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')  # Corrected variable name
    return mfccs





# Create the Streamlit app
st.title('Audio Classification App')

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    st.audio(uploaded_file)

    if st.button('Classify'):
        print_prediction(uploaded_file)

