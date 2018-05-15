import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def extract_raw_driving_log(log_path, correction_steer=0.2):
    csv_lines = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_lines.append(line)
        return csv_lines


def generate_batch_data(raw_samples, batch_size=32, correction_steer=0.2):
    num_samples = len(raw_samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = raw_samples[offset:offset + batch_size]
            images, steer_angles = [], [],
            for sample in batch_samples:
                center_img = cv2.imread('data/IMG/' + sample[0].split('/')[-1])
                left_img = cv2.imread('data/IMG/' + sample[1].split('/')[-1])
                right_img = cv2.imread('data/IMG/' + sample[2].split('/')[-1])
                images.extend([center_img, left_img, right_img])

                center_steer = float(sample[3])
                left_steer = center_steer + correction_steer
                right_steer = center_steer - correction_steer
                steer_angles.extend([center_steer, left_steer, right_steer])
                images, steer_angles = flip_images_steer(images, steer_angles)
            yield shuffle(np.array(images), np.array(steer_angles))


def flip_images_steer(images, steer_angles):
    aug_images, aug_steer_angles = [], []
    for img, angle in zip(images, steer_angles):
        aug_images.append(img)
        aug_steer_angles.append(angle)
        aug_images.append(cv2.flip(img, 1))
        aug_steer_angles.append(angle * -1.0)
    return aug_images, aug_steer_angles


def lenet_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0,  0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda img: (img / 255.0) - 0.5))
    model.add(Convolution2D(filters=24, kernel_size=5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(filters=36, kernel_size=5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(filters=48, kernel_size=5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(filters=64, kernel_size=3, strides=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    print(model.summary())
    return model


def compile_and_train_model(
    model, train_generator, validation_generator, n_train_samples, n_validation_samples,
    save=True,
    loss='mse',
    optimizer='adam',
    epochs=5
):
    model.compile(loss=loss, optimizer=optimizer)
    model.fit_generator(
        train_generator,
        steps_per_epoch=n_train_samples,
        validation_data=validation_generator,
        validation_steps=n_validation_samples,
        epochs=epochs
    )
    if save:
        model.save('model.h5')


if __name__ == '__main__':
    csv_path = './data/driving_log.csv'
    raw_samples = extract_raw_driving_log(csv_path)
    train_samples, validation_samples = train_test_split(raw_samples, test_size=0.2)
    train_generator, validation_generator = generate_batch_data(train_samples), generate_batch_data(validation_samples)
    model = nvidia_model()
    compile_and_train_model(model, train_generator,  validation_generator, len(train_samples), len(validation_samples), epochs=3)



