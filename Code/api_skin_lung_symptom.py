from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import keras.utils as image
from tempfile import NamedTemporaryFile
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.applications import resnet
from symptom_api import recommend_diseases, Sddf, dia, treatments, sym
app = Flask(__name__)
CORS(app)


model_skin = load_model("D:\MSc Data Analytics\Project\Individual Project\Datasets\Skin cancer ISIC The International Skin Imaging Collaboration\Trained Models\skinDiseaseDetectionUsningDenseNet.h5", compile=False)
model_lung = load_model("D:\MSc Data Analytics\Project\Individual Project\Datasets\Lung cancer\Trained Models\Lung_Dense.h5", compile=False)
recognized_indices = []


@app.route('/cancer', methods=['POST'])
def process_cancer_detection():
    prediction_type = request.form.get('type')
    print ('Prediction', prediction_type)
    
    if prediction_type == 'skin':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        return skin_detection(file)
    
    elif prediction_type == 'lung':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        return lung_detection(file)
    
    else:
        return jsonify({'error': 'Invalid prediction type'}), 400
    
    
@app.route('/symptom', methods=['POST'])
def recognize_symptoms():
    try:
        data = request.get_json()
        print(data)

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid JSON format or empty data.'}), 400

        
        for entry in data:
            user_symptom = entry.get('symptom')
            print(user_symptom)

            if user_symptom is not None:
                recognized_index = sym[sym['symptom'] == user_symptom]['syd'].values
                print(recognized_index)
                if len(recognized_index) > 0:
                    recognized_indices.append(recognized_index[0])
            
        results = recommend_diseases(recognized_indices, Sddf, dia, treatments, sym)
        print("recommended diseases", results)

        return jsonify(results)
       
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def skin_detection(file):
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    temp = NamedTemporaryFile(delete=False)
    file.save(temp.name)

    # Process the image for skin detection
    img = image.load_img(temp.name, target_size=(75, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    skin_class = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus',
                  'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

    preds_skin = model_skin.predict(x)[0]
    skin_label_index = np.argmax(preds_skin)
    skin_label = skin_class[skin_label_index]

    # Get the predicted score for the chosen class
    skin_score = preds_skin[skin_label_index]

    message = "Early detection of skin cancer is crucial. Please contact the NHS promptly for evaluation and guidance. Delay could lead to severe health consequences."

    result = {
        'pred': skin_label,
        'score': float(skin_score),
        'message': message
    }
    
    return jsonify({"result": result})

def lung_detection(file):
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    temp = NamedTemporaryFile(delete=False)
    file.save(temp.name)

    # Process the image for lung cancer detection
    img = image.load_img(temp.name, target_size=(460, 460))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    lung_class = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

    preds_lung = model_lung.predict(x)[0]
    lung_label_index = np.argmax(preds_lung)
    lung_label = lung_class[lung_label_index]

    # Get the predicted score for the chosen class
    lung_score = preds_lung[lung_label_index]

     # Check the lung label and provide appropriate messages
    if lung_label == 'normal':
        message = "No Lung Cancer was detected. However, if you want any guidance, feel free to contact NHS."
    else:
        message = "Lung cancer is detected. Please contact the NHS promptly for evaluation and guidance. Delay could lead to severe health consequences."

    result = {
        'pred': lung_label,
        'score': float(lung_score),
        'message': message
    }
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
