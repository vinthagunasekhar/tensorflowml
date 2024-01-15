import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the model
g = Dense(units=1, input_shape=[1])
model = Sequential([g])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Load data form a csv file, We won't expect user giving data manually for large data
file_path= input("Enter the path to your CSV File: ")
data=pd.read_csv(file_path)

# Interactively ask the user for the number of data points
num_points = int(input("How many xs values are you going to provide? "))
# Interactively ask the user for the data points
# xs=[]
# ys=[]
#print("Please enter xs values followed by ys values:")
#for i in range(num_points):
#    x = float(input(f"Enter x value {i+1}: "))
#    y = float(input(f"Enter y value {i+1}: "))
#    xs.append(x)
#    ys.append(y)

# Convert lists to numpy arrays
xs = np.array(data['x'], dtype=float)
ys = np.array(data['y'], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Predict a value
value_to_predict = float(input("Enter a value to predict its corresponding y: "))
print(model.predict([value_to_predict]))

# Print what the model has learned
print("Here is what I learned: {}".format(g.get_weights()))
