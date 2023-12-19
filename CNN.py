from tensorflow import keras
import numpy as np
import pandas as pd
import os
import glob
import imageio.v2 as imageio
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from PIL import Image

image_paths = glob.glob("Dataset\\SampleMoviePosters\\*.jpg")
image_ids = []
for path in image_paths:
    start = path.rfind("/")+1
    end = len(path)-4
    image_ids.append(os.path.splitext(os.path.basename(path))[0])

data = pd.read_csv("Dataset\\MovieGenre.csv")
y = []
classes = tuple()
for image_id in image_ids:
    genres = tuple((data[data["imdbId"] == int(image_id)]["Genre"].values[0]).split("|"))
    y.append(genres)
    classes = classes + genres
mlb = MultiLabelBinarizer()
mlb.fit(y)
y = mlb.transform(y)
classes = set(classes)

def get_image(image_path):
    image = imageio.imread(image_path)
    image = resize(image, (224, 224))
    image = image.astype(np.float32)
    return image
x = []
for path in image_paths:
    x.append(get_image(path))
x = np.asarray(x)
trainx, testx, trainy, testy = train_test_split(x,y,test_size = 0.2, random_state = 42)
print("loaded data")

# Define output classes
output_classes = len(np.unique(genres))

# Scale images to the [0, 1] range
trainx = trainx / 255.0
testy = testy / 255.0

# Make sure images have shape (224, 224, 3)
trainx.reshape(-1, 224, 224, 3)
#trainx = np.expand_dims(trainx, 1)
#testy = testy.reshape(-1, 224, 224, 3)

## Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(output_classes)
])
print("build model")
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainx, trainy, epochs=5, batch_size=32)
print("fitting model")
## Evaluate the trained model
test_loss, test_acc = model.evaluate(testy, testy, verbose=2)
print('\n Test accuracy:', test_acc)
