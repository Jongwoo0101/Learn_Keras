from keras.layers import Input, Dense
from keras.models import Model

# 입력 레이어 정의
inputs = Input(shape=(10,))

# 은닉층 1 정의
hidden1 = Dense(32, activation='relu')(inputs)

# 은닉층 2 정의
hidden2 = Dense(64, activation='relu')(hidden1)

# 출력 레이어 정의
outputs = Dense(1, activation='sigmoid')(hidden2)

# 함수형 API 모델 생성
model = Model(inputs=inputs, outputs=outputs)

# 모델 요약 정보 출력
model.summary()
