from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import *

print (tf.__version__) # Must be v1.1+

model = ResNet50(weights='imagenet')

img = image.load_img('test_images/bike.JPG', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Resnet Predicted:', decode_predictions(preds, top=5)[0])
