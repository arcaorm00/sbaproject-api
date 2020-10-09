import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from util.file_helper import FileReader

class MemberTraining:

    train_data: object = None
    validation_data: object = None
    test_data: object = None
    model: object = None

    def __init__(self):
        self.reader = FileReader()

    def hook(self):
        self.get_data()
        # self.create_model()
        # self.train_model()
        # self.eval_model()
        # self.debug_model()

    @staticmethod
    def create_train(this):
        return this.drop('Exited', axis=1)

    @staticmethod
    def create_label(this):
        return this['Exited']

    def get_data(self):
        self.reader.context = os.path.join(baseurl, 'data_preprocessed')
        self.reader.fname = 'member_preprocessed.csv'
        data = self.reader.csv_to_dframe()
        data = data.to_numpy()
        print(data)

        # x = self.create_train(data)
        # y = self.create_label(data)

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        data = tf.data.Dataset.from_tensor_slices(data)
        self.train_data, self.validation_data, self.test_data = tfds.Split(
            name=data,
            split=('train[:60%]', 'train[60%:]', 'test')
        )
        # 결국 train validation test로 나누고 싶은 것 좀 더 생각해보기
        # num_validation = 7000
        # num_test = 3000
        # self.test_data = data[:num_test]
        # self.validation_data = data[num_test : num_test + num_validation]
        # self.train_data = data[num_test + num_validation :]

        # self.train_data, self.validation_data, self.test_data = tfds.load(
        #     name="imdb_reviews", 
        #     split=('train[:60%]', 'train[60%:]', 'test'),
        #     as_supervised=True
        # )
        print(data)
    
    # 모델 생성 (교과서 p.507)
    # Dense: 완전 연결층
    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # output
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
 
    # 모델 훈련
    def train_model(self):
        self.model.fit(self.train_data.shuffle(7000).batch(512), epochs=20, 
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
    training.hook()
    
    