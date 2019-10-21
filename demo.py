# -*- coding: UTF-8 -*-

import os
import keras
from xlnet_embedding import sentence2idx, idx2sentence, XlnetEmbedding
from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import codecs

class f1_callback(keras.callbacks.Callback):
	def __init__(self, val_data):
		self.x_val = val_data[0]
		self.y_val = val_data[1]

	def on_epoch_end(self, epoch, logs={}):        
		y_pred = self.model.predict(self.x_val, batch_size=model_hyper["batch_size"])
		y_pred = np.argmax(y_pred, axis=1)
		y_val = np.argmax(self.y_val, axis=1)
		result = classification_report(y_val, y_pred)      

		print(result)
		return

# 自行修改需要的配置
def get_config():
	global model_hyper, xlnet_hyper

	model_hyper = {
		'len_max': 30,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
		'label': 10,  # 类别数
		'batch_size': 16,  
		'epochs': 5,  # 训练最大轮次
		'patience': 3, # 早停,2-3就好
		'lr': 5e-5,  # 学习率
		'model_path': './model/model.h5', # 模型保存地址
	}

	xlnet_hyper = {
		# 下载的参数路径
		'model_path': "./chinese_xlnet_base_L-12_H-768_A-12",
		# 微调后保存地址
		'path_fineture': "./model/embedding_trainable.h5",
		# 选择输出的层数 范围 [0, 12(24)], 12或24取决于用的是base还是mid, -1即最后一层 12/24
		'layer_indexes': [-2],       
		'len_max': model_hyper["len_max"],
		'batch_size': model_hyper["batch_size"],
		# 是否微调embedding
		'trainable': True,
		# ['bi', 'uni']
		'attention_type': 'bi',  
		'memory_len': 0,
		# 选择多层输出时处理多层输出的方式： ["add", "avg", "max", "concat"]
		'merge_type': "add"
	}

# 自行修改文本处理方式
def process_data(filename, mode = "train"):
	def process_train():
		nonlocal filename
		f = codecs.open(filename, "r", "UTF-8")
		X, y = [], []
		for line in f:
			line = line.strip().split("\t")
			if len(line)<2:
				continue
			# 双输入要以tuple形式保存   即  X.append( ("text1", "text2") )
			#X.append((line[0].replace(" ", "")[:len(line[0].replace(" ", ""))//2], line[0].replace(" ", "")[len(line[0].replace(" ", ""))//2:]))
			X.append(line[0].replace(" ", ""))
			y.append(int(line[1]))
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
		length = model_hyper["batch_size"]
		print(np.shape(X_train))
		print(np.shape(X_val))
		train_length = (len(y_train)//length) * length
		val_length = (len(y_val)//length) * length
		print(train_length, val_length)
		return X_train[:train_length], X_val[:val_length], y_train[:train_length], y_val[:val_length]

	def process_test():
		nonlocal filename
		f = codecs.open(filename, "r", "UTF-8")
		X, y = [], []
		for line in f:
			line = line.strip().split("\t")
			if len(line)<2:
				continue
			X.append(line[0].replace(" ", ""))
			#X.append((line[0].replace(" ", "")[:len(line[0].replace(" ", ""))//2], line[0].replace(" ", "")[len(line[0].replace(" ", ""))//2:]))
			y.append(int(line[1]))
		print(np.shape(X))
		return X, y

	def process_predict():
		nonlocal filename
		f = codecs.open(filename, "r", "UTF-8")
		X = []
		for line in f:
			line = line.strip()
			X.append(line.replace(" ", ""))
			#X.append((line.replace(" ", "")[:len(line.replace(" ", ""))//2], line.replace(" ", "")[len(line.replace(" ", ""))//2:]))
		return X

	if mode == "train":
		return process_train()
	elif mode == "test":
		return process_test()
	return process_predict()

def encode_data(X, y=None):
	x = []

	for sample in X:
		if type(sample)==tuple and len(sample)==2:
			encoded = sentence2idx(xlnet_hyper["len_max"], sample[0], sample[1])
		else:
			encoded = sentence2idx(xlnet_hyper["len_max"], sample)
		x.append(encoded)

	x_1 = np.array([i[0][0] for i in x])
	x_2 = np.array([i[1][0] for i in x])
	x_3 = np.array([i[2][0] for i in x])
	if xlnet_hyper["trainable"] == True:
		x_all = [x_1, x_2, x_3, np.zeros(np.shape(x_1))]
	else:
		x_all = [x_1, x_2, x_3]

	if y!=None:
		onehot_label = []
		for sample in y:
			onehot = [0] * model_hyper["label"]
			onehot[sample] = 1
			onehot_label.append(onehot)

		onehot_label = np.array(onehot_label)
		return x_all, onehot_label
	return x_all

def create_model():
	
	if embedding.built==False:
		embedding.build()
	emb = embedding.output
	# 自行修改embedding后的模型结构
	# fast text
	x = GlobalAveragePooling1D()(emb)
	output = Dense(model_hyper["label"], activation='softmax')(x)
	model = Model(inputs=embedding.input, outputs=output)
	#model.summary()
	return model

def init():
	global embedding
	get_config()
	embedding = XlnetEmbedding(hyper_parameters=xlnet_hyper)

def train(filename):
	X_train, X_val, y_train, y_val = process_data(filename, mode="train")
	model = create_model()
	encoded_x_train, encoded_y_train = encode_data(X_train, y_train)
	encoded_x_val, encoded_y_val = encode_data(X_val, y_val)

	model.compile(
		optimizer = Adam(lr=model_hyper["lr"], beta_1=0.9, beta_2=0.999, decay=0.0),
		loss = 'categorical_crossentropy',
		metrics = ['accuracy']
	)

	model.fit(
		encoded_x_train, encoded_y_train, 
		batch_size = model_hyper["batch_size"],
		epochs = model_hyper["epochs"], 
		validation_data = (encoded_x_val, encoded_y_val),
		callbacks = [ 
			EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=model_hyper["patience"]),
			ModelCheckpoint(monitor='val_loss', mode='min', filepath=model_hyper["model_path"], verbose=1, save_best_only=True, save_weights_only=True),
			f1_callback(val_data=[encoded_x_val, encoded_y_val])
		]
	)
	# 保存embedding
	if xlnet_hyper["trainable"]:
		embedding.model.save(xlnet_hyper["path_fineture"])


def test(filename):
	global model_hyper
	X_test, y_test = process_data(filename, mode="test")
	model = create_model()
	if os.path.exists(model_hyper["model_path"]):
		model.load_weights(model_hyper["model_path"])
	else:
		raise RuntimeError("model path {} doesn't exist!".format(model_hyper["model_path"]))
	encoded_x_test, encoded_y_test = encode_data(X_test, y_test)

	# batsh_size 可以改大一点，但是必须可以整除测试样本数量
	y_pred = model.predict(encoded_x_test, batch_size = 20)
	y_pred = np.argmax(y_pred, axis=1)
	y_val = np.argmax(encoded_y_test, axis=1)
	result = classification_report(y_val, y_pred)   
	print(result)
	acc = accuracy_score(y_val, y_pred)
	print("acc = {}".format(acc))   


def predict(filename, outfile="predict_result.txt"):
	global model_hyper
	X_pre = process_data(filename, mode="predict")
	model = create_model()
	if os.path.exists(model_hyper["model_path"]):
		model.load_weights(model_hyper["model_path"])
	else:
		raise RuntimeError("model path {} doesn't exist!".format(model_hyper["model_path"]))
	encoded_x_pre = encode_data(X_pre)

	# batsh_size 可以改大一点，但是必须可以整除样本数量
	y_pred = model.predict(encoded_x_pre, batch_size = 1)
	y_pred = np.argmax(y_pred, axis=1)
	outfile = codecs.open(outfile, "w", "UTF-8")
	result = list(zip(X_pre, y_pred))
	for group in result:
		outfile.write("{}\t{}\n".format(group[0], group[1]))
	outfile.close()

if __name__ == '__main__':
	init()
	train("./data/train.txt")
	test("./data/test.txt")
	predict("./data/predict.txt")
