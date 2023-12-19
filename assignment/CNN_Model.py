from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load the dataset
data_dir = "Dataset\\SampleMoviePosters\\"
csv_file = "Dataset/MovieGenre.csv"

df = pd.read_csv(csv_file, encoding='latin-1')
print(df.columns)
df = df[['imdbId', 'Title', 'Genre', 'Poster']]

# Check which columns are present in the DataFrame
df = df.dropna()
df = df.reset_index(drop=True)

# Data preprocessing
img_height = 268
img_width = 182
img_data_array = []
labels = []
image_paths = glob.glob("Dataset\\SampleMoviePosters\\*.jpg")

code_to_index = {'Action': 0, 'Adventure': 1, 'Animation': 2, 'Biography': 3, 'Comedy': 4,
                 'Crime': 5, 'Documentary': 6, 'Drama': 7, 'Family': 8, 'Fantasy': 9,
                 'Film-Noir': 10, 'History': 11, 'Horror': 12, 'Music': 13, 'Musical': 14,
                 'Mystery': 15, 'Romance': 16, 'Sci-Fi': 17, 'Short': 18, 'Sport': 19,
                 'Thriller': 20, 'War': 21, 'Western': 22, 'News': 23, 'Reality-TV': 24,
                 'Talk-Show': 25, 'Game-Show': 26, 'Adult': 27}

for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    img_data_array.append(img)
    
    imdb_id = int(img_path.split('\\')[-1].split('.')[0])
    genre = df.loc[df['imdbId'] == imdb_id]['Genre'].values[0]
    label = np.zeros(28)
    for g in genre.split("|"):
        label[code_to_index[g]] = 1.0
    labels.append(label)

# convert the data into a numpy array
img_data_array = np.array(img_data_array)
labels = np.array(labels)

# Split the dataset
from sklearn.model_selection import train_test_split

# Load and preprocess data
X_train, X_test, y_train, y_test = train_test_split(img_data_array, labels, test_size=0.2, random_state=42)

# Convert y_train to categorical
#y_train_categorical = to_categorical(y_train)

# normalize the pixel values
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# convert labels to categorical
from keras.utils import to_categorical

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

num_classes = 28 # there are 28 genres in our dataset
#IMG_SIZE = (224, 224, 3) # define the input image size



# Build the model
output_classes = 28 # there are 28 genres in our dataset
model = keras.Sequential([
    # Layer 1: Convolution with ReLU activation
    Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    # #Layer 2: Convolution with ReLU activation
    # Conv2D(32, (3, 3)),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2,2)),
    # # Layer 3: Convolution with ReLU activation
    # Conv2D(32, (3, 3)),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2,2)),
    # # Layer 4: Convolution with ReLU activation
    # Conv2D(64, (3, 3)),
    # Activation('relu'),
    # MaxPooling2D(pool_size=(2,2)),
    # Layer 5: Fully Connected Layer with ReLU activation
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    # Layer 6: Fully Connected Layer with ReLU activation
    Dense(32),
    Activation('relu'),
    Dropout(0.5),
    # Layer 7: Fully Connected Layer with Softmax activation
    Dense(num_classes),
    Activation('softmax'),
])
model.summary()
from keras.models import load_model

model.save('Film_CNN_Model.h5')
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128)

# Evaluate the trained model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\n Test accuracy:', test_acc)

