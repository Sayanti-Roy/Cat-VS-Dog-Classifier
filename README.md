# 🐱🐶 Cats vs Dogs Classifier

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with **TensorFlow/Keras**.

---

## 📂 Project Structure.
├── cat_and_dogs/ # Dataset (unpacked)
│ ├── cats_and_dogs_filtered
│ │ ├── train/
│ │ │ ├── cats/
│ │ │ ├── dogs/
│ │ ├── validation/
│ │ ├── cats/
│ │ ├── dogs/
├── cat_dog_classifier.h5 # Trained CNN model (Git LFS)
├── train_model.py # Training script
├── app.py # (Optional) Streamlit app for predictions
├── requirements.txt # Dependencies
└── README.md # Project info


---

## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Debmalya727/Cats-vs-Dogs-Classifier.git
   cd Cats-vs-Dogs-Classifier


Create and activate virtual environment

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Run training (optional if you want to retrain the model)

python train_model.py


Run app (if you have app.py with Streamlit UI)

streamlit run app.py

🧠 Model Details

CNN with:

3 convolutional + max-pooling layers

Fully connected dense layer

Sigmoid output for binary classification

Data augmentation with rotation & horizontal flip

Trained for 100 epochs on Kaggle’s Cats vs Dogs dataset

📊 Results

Validation accuracy: ~85–90% (depending on training run)

Model saved as: cat_dog_classifier.h5

⚡ Future Improvements

Use transfer learning with VGG16 or ResNet

Improve augmentation for robustness

Deploy on Streamlit Cloud or Hugging Face Spaces

📜 License

This project is licensed under the MIT License.


---

👉 You can save this as `README.md` at the root of your repo.  

Do you also want me to generate a **`requirements.txt`** file automatically for you (so others can `pip install -r requirements.txt` easily)?
