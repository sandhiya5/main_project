import pickle
import tensorflow

from flask import Flask,render_template,request
import cv2,os
import numpy as np
import tensorflow as tf
import time

app = Flask(__name__,template_folder='template' )
#print(app.jinja_env.loader.get_source(app, 'index.html'))

def load_pickle(filepath):
  """Loads a pickle from the given filepath."""
  with open(filepath, 'rb') as f:
    model_weights, model_config = pickle.load(f)

  model = tf.keras.models.Model.from_config(model_config)
  model.set_weights(model_weights)

  return model
model = load_pickle(r"model.pkl")

# Function to classify a single image
def classify_single_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Resize the image to match the desired input shape
    resized_img = cv2.resize(img, (224, 224))
    # Preprocess the image
    processed_img = resized_img.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    # Predict probabilities for the positive class
    probability_positive = model.predict(processed_img)
    # Set a threshold for classification
    threshold = 0.5
    # Convert probability to binary prediction
    predicted_class = 'COVID-19 positive' if probability_positive > threshold else 'COVID-19 negative'
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']

        # Check for valid image extension
        allowed_extensions = {'jpg', 'jpeg', 'png'}  # Adjust as needed
        if image.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('error.html', message="Invalid image format")

        # Save the file locally with a unique filename
        filename = f"{int(time.time())}.{image.filename.split('.')[-1]}"  # Use timestamp and extension
        app.config['UPLOAD_FOLDER'] = 'upload_folder'  # Replace with your desired folder name

        
        image_path = os.path.join(r'C:\Users\DELL\OneDrive\Desktop\New folder (3)\upload_folder', filename)
        image.save(image_path)

        # Use the image path for model prediction
        predicted_class = classify_single_image(image_path)
        return render_template('second_page.html', predicted_class=predicted_class)
    return render_template('page1.html')

if __name__ == '_main_':
  app.run(debug=True)