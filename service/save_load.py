import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from util.file_helper import FileReader
 
class SaveLoad:
    
    train_datas: object = None
    train_labels: object = None
    test_datas: object = None
    test_labels: object = None
 
    def __init__(self):
        self.reader = FileReader()
 
    def hook(self):
        self.get_data()
        self.create_model()
        self.train_model()
        # self.save_model()
 
    def get_data(self):
        print(f'baseurl: {baseurl}')
        self.reader.context = os.path.join('C:/Users/saltQ/sbaproject-api/model', 'data_preprocessed')
        self.reader.fname = 'member_preprocessed.csv'
        data = self.reader.csv_to_dframe()
        data = data.to_numpy()
        print(data[:60])

        table_col = data.shape[1]
        y_col = 1
        x_col = table_col - y_col
        x = data[:, 0:x_col]
        y = data[:, x_col:]

        train_datas, test_datas, train_labels, test_labels = train_test_split(x, y, test_size=0.4)

        self.train_labels = train_labels[:1000]
        self.test_labels = test_labels[:1000]
        self.train_datas = train_datas[:1000]
        self.test_datas = test_datas[:1000]
 
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
    def train_model(self):
        checkpoint_path = 'training_1/cp.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        print('***** fit *****')
        #  ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 784 but received input with shape [None, 12]
        self.model.fit(x=self.train_datas, y=self.train_labels, epochs=10, 
        validation_data = (self.test_datas, self.test_labels), callbacks=[cp_callback]) # 훈련 단계 콜백 전달
        self.model.load_weights(checkpoint_path) # 가중치 추가
        loss, acc = self.model.evaluate(self.test_datas, self.test_labels, verbose=2)
        # verbose는 학습 진행상황을 보여줄지 말지에 대한 옵션
        print('복원된 모델의 정확도: {:5.2f}%' .format(100 * acc))
        # 파일 이름에 에포크 번호를 포함시킨다.
        checkpoint_path = os.path.join(baseurl, 'training_2', 'cp-{epoch: 04d}.ckpt')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose = 1, save_weights_only=True,
            period = 5 # 5번째 에포크마다 가중치를 저장한다.  
        )
        print(f'checkpoint: {checkpoint_path}')
        self.model.save_weights(checkpoint_path.format(epoch=0))
        self.model.fit(self.train_datas, self.train_labels, epochs=50, 
        callbacks=[cp_callback], validation_data=(self.test_datas, self.test_labels), verbose=0)
 
    # 전체 모델을 HDF5 파일로 저장한다.
    def save_model(self):
        context = os.path.join(baseurl, 'saved_model')
        self.model.save(os.path.join(context, 'my_model.h5'))
        print('=' * 30)
    
    def load_model(self):
        self.new_model = keras.models.load_model('my_model.h5')
        self.new_model.summary()
        loss, acc = self.new_model.evaluate(self.test_images, self.test_labels, verbose = 2)
 
    def debug_model(self):
        print(f'모델정보: {self.model.summary()}')
 
if __name__ == '__main__':
    api = SaveLoad()
    api.hook()
