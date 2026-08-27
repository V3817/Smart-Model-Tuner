# Smart Model Tuner

Smart Model Tuner is a Streamlit web app for fine-tuning language models via the Lamini API. The app allows you to manage datasets, adjust training parameters, and monitor the fine-tuning process in real time.

## Features

- **Dataset Management:**  
  Upload a CSV or manually input training examples. The app performs quality checks and provides dataset statistics.
  
- **Hyperparameter Tuning:**  
  Configure core training parameters like learning rate, epochs, and batch size, with options for early stopping.
  
- **Training Visualization:**  
  Get real-time status updates during training and view a summary model card with key metrics once fine-tuning is complete.

## Requirements

- Python 3.7 or higher  
- Streamlit  
- Lamini Python Client  
- Pandas

## Installation

Install the required packages using pip:

pip install streamlit lamini pandas

## Usage

1. Run the app:

   streamlit run streamlit_tuner.py

2. Enter your Lamini API key in the sidebar.
3. Upload or input your training data.
4. Adjust hyperparameters and resource settings as needed.
5. Click "Start Optimized Training" to begin the fine-tuning process.

## License

This project is licensed under the MIT License.
