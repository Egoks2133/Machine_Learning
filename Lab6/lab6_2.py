import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def preprocess_image(image_path, img_size=(150, 150)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    model = load_model('helmet_detection_model.h5')

    image_path = input("Введите путь к изображению: ")

    img_array = preprocess_image(image_path, img_size=(150, 150))

    prediction = model.predict(img_array, verbose=0)

    probability_no_helmet = prediction[0][0]
    probability_with_helmet = 1 - probability_no_helmet

    if probability_no_helmet > 0.5:
        predicted_class = "Без шлема"
        confidence = probability_no_helmet
    else:
        predicted_class = "С шлемом"
        confidence = probability_with_helmet

    print(f"Результат классификации: {predicted_class}")
    print(f"Уверенность: {confidence:.2%}")
    print(f"Вероятность 'С шлемом': {probability_with_helmet:.4f} ({probability_with_helmet:.2%})")
    print(f"Вероятность 'Без шлема': {probability_no_helmet:.4f} ({probability_no_helmet:.2%})")


if __name__ == '__main__':
    main()