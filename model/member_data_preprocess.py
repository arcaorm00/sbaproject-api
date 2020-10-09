import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
baseurl = os.path.dirname(os.path.abspath(__file__))
from util.file_helper import FileReader
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

'''
RowNumber,        레코드 행 번호
CustomerId,       고객 ID
Surname,          고객 last name
CreditScore,      신용점수
Geography,        지역
Gender,           성별
Age,              나이
Tenure,           존속 기간
Balance,          잔액
NumOfProducts,    구매 물품 수
HasCrCard,        신용카드 여부
IsActiveMember,   활성 고객 여부
EstimatedSalary,  급여 수준
Exited            서비스 탈퇴 여부
'''

# DB 저장용 파일: member_detail.csv
# memberList 비롯 get은 DB에서 받아온 정보를 바로 바로 정제할 예정: 일단은 member_preprocessed.py로 저장
class MemberDataPreprocess:

    def __init__(self):
        self.filereader = FileReader()

    def hook_process(self):
        this = self.filereader
        this.context = os.path.join(baseurl, 'data')
        # 데이터 정제 전 raw data
        this.fname = 'member_detail.csv'
        members = this.csv_to_dframe()
        this.train = members
        
        # 칼럼 삭제
        this = self.drop_feature(this, 'RowNumber') # 열 번호 삭제
        this = self.drop_feature(this, 'Surname') # 이름 삭제
        this = self.drop_feature(this, 'Email') # 이메일 삭제
        this = self.drop_feature(this, 'Role') # 권한 삭제
        this = self.drop_feature(this, 'Password') # 비밀번호 삭제
        
        # 데이터 정제
        this = self.geography_nominal(this)
        this = self.gender_nominal(this)
        this = self.age_ordinal(this)
        this = self.drop_feature(this, 'Age')
        this = self.creditScore_ordinal(this)
        this = self.balance_ordinal(this)
        this = self.estimatedSalary_ordinal(this)

        # 고객의 서비스 이탈과 각 칼럼간의 상관계수
        self.correlation_member_secession(this.train)

        # 정제 데이터 저장
        # self.save_preprocessed_data(this)

        # label 컬럼 재배치
        this = self.columns_relocation(this)

        # 훈련 데이터, 레이블 데이터 분리
        # this.label = self.create_label(this)
        # this.train = self.create_train(this)
        
        print(this)
        

    # 고객의 서비스 이탈과 각 칼럼간의 상관계수
    def correlation_member_secession(self, members):
        member_columns = members.columns
        member_correlation = {}
        for col in member_columns:
            cor = np.corrcoef(members[col], members['Exited'])
            # print(cor)
            member_correlation[col] = cor
        print(member_correlation)
        '''
        r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
        r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
        r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
        r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
        r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
        r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
        r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계
        {'CustomerId': array([[ 1.        , -0.00624799], [-0.00624799,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'CreditScore': array([[ 1.        , -0.02709354], [-0.02709354,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Geography': array([[1.        , 0.15377058], [0.15377058, 1.        ]]), ==> 약한 양적 선형관계
        'Gender': array([[1.        , 0.10651249], [0.10651249, 1.        ]]), ==> 약한 양적 선형관계
        'Age': array([[1.        , 0.28532304], [0.28532304, 1.        ]]), ==> 약한 양적 선형관계
        'Tenure': array([[ 1.        , -0.01400061], [-0.01400061,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Balance': array([[1.        , 0.11853277], [0.11853277, 1.        ]]), ==> 약한 양적 선형관계
        'NumOfProducts': array([[ 1.        , -0.04781986], [-0.04781986,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'HasCrCard': array([[ 1.        , -0.00713777], [-0.00713777,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'IsActiveMember': array([[ 1.        , -0.15612828], [-0.15612828,  1.        ]]), ==> 약한 음적 선형관계
        'EstimatedSalary': array([[1.        , 0.01300995], [0.01300995, 1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'Exited': array([[1., 1.], [1., 1.]]), 
        'AgeGroup': array([[1.        , 0.21620629], [0.21620629, 1.        ]])} ==> 약한 양적 선형관계
        '''

    # ---------------------- 데이터 정제 ----------------------
    @staticmethod
    def create_train(this):
        return this.train.drop('Exited', axis=1)

    @staticmethod
    def create_label(this):
        return this.train['Exited']

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def surname_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        this.train['CreditScore'] = pd.qcut(this.train['CreditScore'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def geography_nominal(this):
        # print(this.train['Geography'].unique()) 
        # ==> ['France' 'Spain' 'Germany']
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        this.train['Geography'] = this.train['Geography'].map(geography_mapping)
        return this

    @staticmethod
    def gender_nominal(this):
        gender_mapping = {'Male': 0, 'Female': 1}
        this.train['Gender'] = this.train['Gender'].map(gender_mapping)
        this.train = this.train
        return this

    @staticmethod
    def age_ordinal(this):
        train = this.train
        train['Age'] = train['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 범위
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'YoungAdult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Baby', 
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'YoungAdult',
            6: 'Adult',
            7: 'Senior'
        }

        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1, 
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'YoungAdult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        this.train = train
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
        this.train['Balance'] = pd.qcut(this.train['Balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def numOfProducts_ordinal(this):
        return this

    @staticmethod
    def hasCrCard_numeric(this):
        return this

    @staticmethod
    def isActiveMember_numeric(this):
        return this

    @staticmethod
    def estimatedSalary_ordinal(this):
        this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    # ---------------------- 파일 저장 ---------------------- 
    def save_preprocessed_data(self, this):
        this.context = os.path.join(baseurl, 'data_preprocessed')
        this.train.to_csv(os.path.join(this.context, 'member_preprocessed.csv'))

    # ---------------------- label 컬럼 위치 조정 ---------------------- 
    def columns_relocation(self, this):
        cols = this.train.columns.tolist()
        # ['CustomerId', 'CreditScore', 'Geography', 'Gender', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'AgeGroup']
        cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
        # ['CustomerId', 'CreditScore', 'Geography', 'Gender', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'AgeGroup', 'Exited']
        this.train = this.train[cols]
        return this
    

    # ---------------------- 모델 훈련 ---------------------- 

if __name__ == '__main__':
    member = MemberDataPreprocess()
    member.hook_process()
