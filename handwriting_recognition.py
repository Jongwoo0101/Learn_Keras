from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 모델 구조 정의
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 모델 컴파일
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 훈련 및 테스트 데이터 전처리
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 레이블을 범주형 형식으로 변환
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 모델 훈련
network.fit(train_images, train_labels, epochs=5, batch_size=128)




