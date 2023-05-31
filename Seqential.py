import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤한 입력 데이터 생성
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# 모델 구축
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=10))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

