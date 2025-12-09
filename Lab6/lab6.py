import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import kagglehub


def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    classes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip().lower()
        classes.append(class_name)

    has_without_helmet = any('without' in cls for cls in classes)
    has_with_helmet = any('with' in cls and 'without' not in cls for cls in classes)

    if has_without_helmet:
        return 1  # Без шлема
    elif has_with_helmet:
        return 0  # С шлемом
    else:
        return None


def load_dataset(dataset_path, img_size=(150, 150), max_samples=2000):
    images = []
    labels = []

    images_dir = os.path.join(dataset_path, "images")
    annotations_dir = os.path.join(dataset_path, "annotations")

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i, img_file in enumerate(image_files[:max_samples]):
        xml_file = os.path.splitext(img_file)[0] + ".xml"
        xml_path = os.path.join(annotations_dir, xml_file)

        if not os.path.exists(xml_path):
            continue

        label = parse_xml_annotation(xml_path)
        if label is None:
            continue

        img_path = os.path.join(images_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            continue

    return np.array(images), np.array(labels)


def main():
    path = kagglehub.dataset_download("andrewmvd/helmet-detection")

    images, labels = load_dataset(path, img_size=(150, 150), max_samples=2000)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_images, train_labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nТочность на тестовой выборке: {test_acc:.4f}")

    model.save('helmet_detection_model.h5')


if __name__ == '__main__':
    main()