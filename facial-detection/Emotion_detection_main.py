from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
import keras
from keras.preprocessing.image import ImageDataGenerator

# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
base_model = ResNet50(weights = 'imagenet', include_top=False, pooling = 'avg')

# 构建网络的最后一层，3是自己的数据的类别
SN_fcl = Dense(2)(base_model.output)

# 定义整个模型
Feature_extract_model = Model(inputs=base_model.input, outputs=predictions)

