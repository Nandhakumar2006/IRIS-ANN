import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("iris_ann_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled = scaler.transform(input_data)
    pred = np.argmax(model.predict(scaled), axis=1)
    species = le.inverse_transform(pred)[0]
    return f"🌸 Predicted Species: {species}"


interface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Number(label="🌿 Sepal Length (cm)"),
        gr.Number(label="🌿 Sepal Width (cm)"),
        gr.Number(label="🌸 Petal Length (cm)"),
        gr.Number(label="🌸 Petal Width (cm)")
    ],
    outputs=gr.Textbox(label="🧠 Model Prediction"),
    title="🌼 Iris Flower Classifier (ANN)",
    description=(
        "A beautifully designed AI model that predicts the *Iris flower species* "
        "based on four floral measurements. <br><br>"
        "💡 Built with TensorFlow & Gradio · Designed by Nandhakumar."
    ),
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
    allow_flagging="never"
)

interface.launch(inbrowser=True)