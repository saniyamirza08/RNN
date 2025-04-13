import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# Clean up old model file if it exists
if os.path.exists("simple_rnn.h5"):
    os.remove("simple_rnn.h5")

# Load IMDB dataset
max_features = 10000  # vocabulary size
max_len = 500         # max length of a review

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(SimpleRNN(units=128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[earlystop])

# Save model (this version does NOT include time_major)
model.save("simple_rnn.h5")
print("✅ Model saved as simple_rnn.h5")
