# Image Classification Using CNN

## Overview
This project demonstrates image classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to classify images of 10 animal classes using the "Animals-10" dataset.

## Features
- Preprocessing of training and test data using `ImageDataGenerator`.
- Data augmentation for robust training.
- CNN architecture with:
  - Convolutional and pooling layers for feature extraction.
  - Fully connected layers for classification.
- Performance evaluation using accuracy and other metrics.
- Option to save and load the trained model for reuse.

## Dataset
- **Dataset:** [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10)
- The dataset includes images of 10 animal categories: dog, cat, horse, butterfly, chicken, sheep, cow, squirrel, elephant, and spider.

## Folder Structure
```
project_directory/
|-- dataset/
|   |-- training_set/
|   |   |-- dog/
|   |   |-- cat/
|   |   |-- ... (other animal folders)
|   |
|   |-- test_set/
|   |   |-- dog/
|   |   |-- cat/
|   |   |-- ... (other animal folders)
|
|-- trained_model.h5
|-- main.py
|-- README.md
```

The `training_set` and `test_set` folders contain subfolders for each class, where the images are stored.

## Setup

### Requirements
- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Running the Code

#### 1. Train the Model
To train the model, run:
```bash
python main.py
```
This will preprocess the dataset, train the CNN, and save the model as `trained_model.h5`.

#### 2. Skip Training and Load the Model
If the model is already trained, load it using:
```python
from tensorflow.keras.models import load_model
cnn = load_model('trained_model.h5')
```

#### 3. Evaluate the Model
Evaluate the model's performance:
```python
results = cnn.evaluate(test_set)
print(f"Test Accuracy: {results[1] * 100:.2f}%")
```

#### 4. Make Predictions
To predict an image:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
test_image = image.load_img('dataset/single_prediction/image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make prediction
result = cnn.predict(test_image)
classes = ['dog', 'cat', 'horse', 'butterfly', 'chicken', 'sheep', 'cow', 'squirrel', 'elephant', 'spider']
prediction = classes[np.argmax(result)]
print(f"Prediction: {prediction}")
```

## Model Architecture
1. **Convolution Layers**
   - Extract features using 32 filters of size 3x3.
2. **Max Pooling Layers**
   - Reduce spatial dimensions.
3. **Flattening**
   - Convert feature maps into a single vector.
4. **Fully Connected Layers**
   - Dense layers with ReLU activation for hidden layers and Sigmoid activation for binary classification.

## Saving and Loading the Model
- **Saving:**
  ```python
  cnn.save('trained_model.h5')
  ```
- **Loading:**
  ```python
  cnn = load_model('trained_model.h5')
  ```

## Evaluation Metrics
- **Accuracy:** Measures the percentage of correctly classified images.
  ```python
  from sklearn.metrics import accuracy_score
  results = cnn.evaluate(test_set)
  print(f"Test Accuracy: {results[1] * 100:.2f}%")
  ```

## Future Enhancements
- Add more classes or datasets for improved generalization.
- Implement data augmentation strategies.
- Experiment with different architectures and hyperparameters.
- Deploy the model using a web interface.

## Author
[Dhawal Phalak](https://github.com/Dhawal1133)

## License
This project is licensed under the MIT License.
