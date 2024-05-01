import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
loaded_model = joblib.load('xgboost_model.pkl')

# Define the predictor variable names
Predictors = ['carat', 'x', 'y', 'z', 'cut', 'color', 'clarity']

# Define label encoders for categorical features
label_encoder_cut = LabelEncoder()
label_encoder_color = LabelEncoder()
label_encoder_clarity = LabelEncoder()

# Function to preprocess input data
def preprocess_input(carat, x, y, z, cut, color, clarity):
    # Convert numerical features to float
    carat = float(carat)
    x = float(x)
    y = float(y)
    z = float(z)
    
    # Fit and transform categorical features
    cut_encoded = label_encoder_cut.fit_transform([cut])[0]
    color_encoded = label_encoder_color.fit_transform([color])[0]
    clarity_encoded = label_encoder_clarity.fit_transform([clarity])[0]

    return [carat, x, y, z, cut_encoded, color_encoded, clarity_encoded]

# Function to make predictions
def predict_price():
    try:
        # Get input values from entry widgets
        carat = carat_entry.get()
        x = x_entry.get()
        y = y_entry.get()
        z = z_entry.get()
        cut = cut_entry.get()
        color = color_entry.get()
        clarity = clarity_entry.get()

        # Preprocess input data
        input_data = preprocess_input(carat, x, y, z, cut, color, clarity)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(pd.DataFrame([input_data], columns=Predictors))

        # Display prediction in a message box
        messagebox.showinfo("Prediction Result", f"The predicted price is: {prediction[0]}")
    except Exception as e:
        # Display error message if any exception happens
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Diamond Price Prediction")

# Input
tk.Label(root, text="Carat:").grid(row=0, column=0, padx=10, pady=5)
carat_entry = tk.Entry(root)
carat_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="X:").grid(row=1, column=0, padx=10, pady=5)
x_entry = tk.Entry(root)
x_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Y:").grid(row=2, column=0, padx=10, pady=5)
y_entry = tk.Entry(root)
y_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Z:").grid(row=3, column=0, padx=10, pady=5)
z_entry = tk.Entry(root)
z_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Cut:").grid(row=4, column=0, padx=10, pady=5)
cut_entry = tk.Entry(root)
cut_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Color:").grid(row=5, column=0, padx=10, pady=5)
color_entry = tk.Entry(root)
color_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Clarity:").grid(row=6, column=0, padx=10, pady=5)
clarity_entry = tk.Entry(root)
clarity_entry.grid(row=6, column=1, padx=10, pady=5)

# Prediction Button
predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()



