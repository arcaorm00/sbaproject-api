import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_hub as hub

class MemberTraining:

    train_data: object = None
    validation_data: object = None
    test_data: object = None
    model: object = None

    def __init__(self):
        pass

    def get_data(self):
        # self.train_data, self.validation_data, self.test_data = tfds.load(
        #     name="imdb_reviews", 
        #     split=('train[:60%]', 'train[60%:]', 'test'),
        #     as_supervised=True
        # )
 
    # 샘플 생성
    def create_sample(self):
        trian_example_batch, train_labels_batch = next(iter(self.train_data.batch(10)))
        return trian_example_batch
    
    # 모델 생성 (교과서 p.507)
    # Dense: 완전 연결층
    def create_model(self):
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
        hub_layer(self.create_sample()[:3])
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # output
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
 
    # 모델 훈련
    def train_model(self):
        self.model.fit(self.train_data.shuffle(10000).batch(512), epochs=20, 
        validation_data=self.validation_data.batch(512), verbose=1) # 512 = 2 ^9
    
    # 모델 평가
    def eval_model(self):
        results = self.model.evaluate(self.test_data.batch(512), verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print('%s: %.3f' % (name, value))
 
    # 모델 디버깅
    def debug_model(self):
        print(f'self.train_data: {self.train_data}')
        print(f'self.validation_data: {self.validation_data}')
        print(f'self.test_data: {self.test_data}')


if __name__ == '__main__':
    training = MemberTraining()
    
    