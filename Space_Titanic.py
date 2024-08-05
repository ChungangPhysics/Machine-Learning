import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 데이터 불러오기
train_df = pd.read_csv('train.csv')

# 결측값 처리 (train 데이터)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['RoomService'] = train_df['RoomService'].fillna(train_df['RoomService'].mean())
train_df['FoodCourt'] = train_df['FoodCourt'].fillna(train_df['FoodCourt'].mean())
train_df['ShoppingMall'] = train_df['ShoppingMall'].fillna(train_df['ShoppingMall'].mean())
train_df['Spa'] = train_df['Spa'].fillna(train_df['Spa'].mean())
train_df['VRDeck'] = train_df['VRDeck'].fillna(train_df['VRDeck'].mean())

train_df['HomePlanet'] = train_df['HomePlanet'].fillna('Unknown')
train_df['CryoSleep'] = train_df['CryoSleep'].fillna(False).infer_objects()
train_df['Cabin'] = train_df['Cabin'].fillna('Unknown')
train_df['Destination'] = train_df['Destination'].fillna('Unknown')
train_df['VIP'] = train_df['VIP'].fillna(False).infer_objects()

# 데이터 타입 변환 (train 데이터)
train_df['CryoSleep'] = train_df['CryoSleep'].astype(bool)
train_df['VIP'] = train_df['VIP'].astype(bool)

# 필요 없는 열 제거 (train 데이터)
train_df.drop(columns=['Name', 'PassengerId'], inplace=True, errors='ignore')

# 범주형 데이터 인코딩 (train 데이터)
label_encoder = LabelEncoder()
train_df['HomePlanet'] = label_encoder.fit_transform(train_df['HomePlanet'])
train_df['Cabin'] = label_encoder.fit_transform(train_df['Cabin'])
train_df['Destination'] = label_encoder.fit_transform(train_df['Destination'])

# 정규화 (train 데이터)
scaler = MinMaxScaler()
train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])

# 특징과 라벨 분리 (train 데이터)
X_train = train_df.drop(columns=['Transported'])
y_train = train_df['Transported']

# hyperparameter 후보 설정
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=200),
                           param_grid=param_grid,
                           cv=5,
                           verbose=2)

# 모델 훈련
grid_search.fit(X_train, y_train)

# 최적의 파라미터 출력
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# 최적의 파라미터로 모델 설정
best_model = grid_search.best_estimator_

# test 데이터 전처리 함수 정의
def preprocess_test_data(test_df, label_encoder, scaler):
    test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
    test_df['RoomService'] = test_df['RoomService'].fillna(test_df['RoomService'].mean())
    test_df['FoodCourt'] = test_df['FoodCourt'].fillna(test_df['FoodCourt'].mean())
    test_df['ShoppingMall'] = test_df['ShoppingMall'].fillna(test_df['ShoppingMall'].mean())
    test_df['Spa'] = test_df['Spa'].fillna(test_df['Spa'].mean())
    test_df['VRDeck'] = test_df['VRDeck'].fillna(test_df['VRDeck'].mean())

    test_df['HomePlanet'] = test_df['HomePlanet'].fillna('Unknown')
    test_df['CryoSleep'] = test_df['CryoSleep'].fillna(False).infer_objects()
    test_df['Cabin'] = test_df['Cabin'].fillna('Unknown')
    test_df['Destination'] = test_df['Destination'].fillna('Unknown')
    test_df['VIP'] = test_df['VIP'].fillna(False).infer_objects()

    test_df['CryoSleep'] = test_df['CryoSleep'].astype(bool)
    test_df['VIP'] = test_df['VIP'].astype(bool)

    test_df.drop(columns=['Name'], inplace=True, errors='ignore')

    # train 데이터에 있는 범주형 값만 인코딩
    for column in ['HomePlanet', 'Cabin', 'Destination']:
        test_df[column] = test_df[column].apply(lambda x: x if x in label_encoder.classes_ else 'Unknown')

    test_df['HomePlanet'] = label_encoder.transform(test_df['HomePlanet'])
    test_df['Cabin'] = label_encoder.transform(test_df['Cabin'])
    test_df['Destination'] = label_encoder.transform(test_df['Destination'])

    test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.transform(test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])

    return test_df

# CSV 파일을 읽어 데이터프레임 생성 (test 데이터)
test_df = pd.read_csv('test.csv')

# test 데이터 전처리
test_df = preprocess_test_data(test_df, label_encoder, scaler)

# 특징 추출 (test 데이터)
X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

# 예측
y_pred = best_model.predict(X_test)

# 결과 저장
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': y_pred})
output.to_csv('prediction_results.csv', index=False)
