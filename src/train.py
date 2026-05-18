import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = "data/chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        horizontal_flip=True,
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        validation_split=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def build_model():
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    train_gen, val_gen, test_gen = get_data_generators()
    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint("models/best_model.keras", monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    preds = model.predict(test_gen)
    pred_labels = (preds > 0.5).astype(int).reshape(-1)
    true_labels = test_gen.classes

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()