# Keras implementation of Google Wide&Deep Learning

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler

import code

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", 
    "hours_per_week", "native_country", "income_bracket"
]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship", 
    "race", "gender", "native_country"
]

CONTINUOUS_COLUMNS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

def load(filename, skiprows=0):
	df = pd.read_csv(
			filename, names=COLUMNS, skipinitialspace=True, skiprows=skiprows, engine='python'
	)
	df = df.dropna(how='any', axis=0)
	return df

def preprocess(df):

	df[LABEL_COLUMN] = df['income_bracket'].apply(lambda x: '50K' in x).astype(int)
	df.pop('income_bracket')
	y = df[LABEL_COLUMN].values
	df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)
	df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
	X = df.values
	return X, y

if __name__ == '__main__':
	df_train = load('data/adult.data')
	df_test = load('data/adult.test', skiprows=1)
	df = pd.concat([df_train, df_test])
	train_len = len(df_train)
	
	X, y = preprocess(df)
	X_train = X[:train_len]
	y_train = y[:train_len]
	X_test = X[train_len:]
	y_test = y[train_len:]

	# wide
	wide = Sequential()
	wide.add(Dense(1, input_dim=X_train.shape[1]))

	# deep
	deep = Sequential()
	deep.add(Dense(input_dim=X_train.shape[1], output_dim=100, activation='relu'))
	deep.add(Dense(input_dim=100, output_dim=32, activation='relu'))
	deep.add(Dense(input_dim=32, output_dim=8))
	deep.add(Dense(1, activation='sigmoid'))

	# wide & deep
	model = Sequential()
	model.add(Merge([wide, deep], mode='concat', concat_axis=1))
	model.add(Dense(1, activation='sigmoid'))

	# compile
	model.compile(
		loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	# train
	model.fit([X_train, X_train], y_train, nb_epoch=10, batch_size=32)

	# eval
	loss, accuracy = model.evaluate([X_test, X_test], y_test)
	print('acc:', accuracy)
