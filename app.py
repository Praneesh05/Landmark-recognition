
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from PIL import Image


@st.cache_data
def load_labels():
    labels_path = tf.keras.utils.get_file(
        'landmarks_classifier_asia_V1_label_map.csv',
        'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
    )
    df = pd.read_csv(labels_path)
    return dict(zip(df['id'], df['name']))


@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1"
    return hub.KerasLayer(model_url, output_key='predictions:logits')

labels = load_labels()
classifier = load_model()


def get_coordinates(location):
    geolocator = Nominatim(user_agent="landmark_app")
    queries = [location + ", Asia", location + ", India", location]
    for query in queries:
        loc = geolocator.geocode(query)
        if loc:
            return loc.latitude, loc.longitude
    return None


def predict_landmark(img: Image.Image):
    img_resized = img.resize((321, 321)).convert('RGB')
    img_arr = np.array(img_resized) / 255.0
    img_arr = img_arr[np.newaxis, ...]
    logits = classifier(img_arr)
    probs = tf.nn.softmax(logits)
    pred_idx = int(np.argmax(probs))
    label = labels.get(pred_idx, "Unknown")
    return label


st.title("🗺️ Asian Landmark Recognition")
st.write("Upload an image and let the model predict the landmark.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        try:
            prediction = predict_landmark(image)
            st.success(f"**Predicted Landmark**: {prediction}")

            coords = get_coordinates(prediction)
            if coords:
                lat, lon = coords
                st.map(pd.DataFrame([[lat, lon]], columns=['lat', 'lon']))
                st.info(f"**Coordinates**: Latitude = {lat}, Longitude = {lon}")
            else:
                st.warning("Could not find coordinates for this landmark.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
