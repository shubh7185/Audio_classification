import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('C:/Users/shubh/OneDrive/Pictures/SQL/audioclassification/audio_classification_model2.h5')
class_names = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

def print_prediction(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    max_pad_len = 174  # Adjust as necessary
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    prediction_feature = mfccs.reshape(1, 40, -1, 1)  # Adjust the shape according to your model's input

    predicted_vector = model.predict(prediction_feature)
    predicted_class_indices = np.argmax(predicted_vector, axis=1)
    predicted_classes = [class_names[index] for index in predicted_class_indices]

    results = {class_names[i]: float(predicted_vector[0][i]) for i in range(len(predicted_vector[0]))}
    return predicted_classes, results



def main():
    st.title('Audio Classification App')
    st.write("Upload an audio file for classification.")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button('Classify'):
            predicted_classes, results = print_prediction(uploaded_file)
            st.write("Predicted Class:", predicted_classes[0])
            st.write("Predicted Probabilities:")
            for label, prob in results.items():
                st.write(f"{label}: {prob}")


if __name__ == '__main__':
    main()



