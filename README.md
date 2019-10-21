## XLNet Embedding



将XLNet作为Embedding的Keras封装，根据需要取出某层或某些层的输出作为特征，并可以在后面搭建自定义的网络（如Fasttext）

### Usage：

1. 下载 XLNet模型：https://github.com/ymcui/Chinese-PreTrained-XLNet

2. 下载代码，解压XLNet模型至代码目录

3. 准备训练数据并放置在data目录

4. 修改配置和网络
5. 训练 / 测试 / 预测

### 代码说明

demo默认任务为文本分类，若目标为其他任务需要自行修改demo.py文件

#### 高频修改函数：

get_config():  模型及XLNet配置

process_data(): 修改文本读取、预处理

create_model(): 在XLNet后增加自己的网络结构，默认为fasttext

#### 中频修改函数：

train(): 训练模型，可在这里修改优化器、回调函数等

test(): 加载训练保存的模型进行测试，使用classification_report 和 accuracy_score， 其他任务可自行修改

predict(): 加载模型进行预测，保存到文件中

#### 不建议修改函数：

encode_data(): 对输入进行编码

init()：初始化参数



### 参考/致谢

1.  **Chinese-PreTrained-XLNet** (ymcui) https://github.com/ymcui/Chinese-PreTrained-XLNet
2.  **keras-xlnet** (CyberZHG)  https://github.com/CyberZHG/keras-xlnet
3.  **Keras-TextClassification** (yongzhuo) https://github.com/yongzhuo/Keras-TextClassification
4.  **xlnet** (zihangdai) https://github.com/zihangdai/xlnet