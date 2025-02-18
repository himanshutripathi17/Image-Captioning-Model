import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pickle

# Load the trained caption model and tokenizer
@st.cache_resource
def load_caption_model():
    return load_model("my_model.keras")

@st.cache_resource
def load_tokenizer():
    with open('working/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

@st.cache_resource
def load_vgg_model():
    vgg = VGG16()
    return Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

caption_model = load_caption_model()
tokenizer = load_tokenizer()
vgg_model = load_vgg_model()

# Constants
MAX_LENGTH = 35  # Replace with your actual max_length
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

# Convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Updated function for predicting captions:
def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'
    for i in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence to the correct length (max_length = 35)
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get index with the highest probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word is not found
        if word is None:
            break
        # Append the word to the input sequence for generating the next word
        in_text += " " + word
        # Stop if we reach the end tag
        if word == 'endseq':
            break

    # Clean the generated caption by removing 'startseq' and 'endseq'
    final_caption = in_text.split()[1:-1]  # Remove 'startseq' and 'endseq'
    return ' '.join(final_caption)  # Return caption with words joined by spaces

# Streamlit UI
st.title("Image Caption Generator")
st.write("Upload an image, and the app will generate a descriptive caption!")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and predict caption
    st.write("Generating caption...")
    processed_image = preprocess_image(image)
    features = vgg_model.predict(processed_image, verbose=0)
    caption = predict_caption(caption_model, features, tokenizer, MAX_LENGTH)

    st.write("### Predicted Caption:")
    st.write(caption)  # Output the caption directly
