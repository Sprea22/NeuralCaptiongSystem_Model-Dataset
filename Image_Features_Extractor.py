from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Input image
file_dir = 'Dataset/Data chart images/Final dataset plots single numbered/'
filename = '1.png'
img_path = file_dir + filename
img = image.load_img(img_path, target_size=(224, 224))

# Input model
model_1000 = vgg16.VGG16(weights='imagenet', include_top=True)

# Preprocessing on the input image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Model prediction: vector size 1000
features_1000 = model_1000.predict(x)

# Model prediction: vector size 4096
model_4096 = Model(input=model_1000.input, output=model_1000.get_layer('fc2').output)
features_4096 = model_4096.predict(x)
features_4096 = features_4096.reshape((4096,1))

print(features_1000.flatten().shape)
print(features_4096.flatten().shape)

#np.savetxt('fc2.txt',fc2_features)
