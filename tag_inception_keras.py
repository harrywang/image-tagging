from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *

print (tf.__version__) # Must be v1.1+

model = InceptionV3(weights='imagenet')

img = image.load_img('test_images/bike.JPG', target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Inception Predicted:', decode_predictions(preds, top=5)[0])
