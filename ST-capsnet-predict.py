from keras import layers, models, optimizers, regularizers, constraints
from keras import backend as K
from capsulelayer_keras import Class_Capsule, Conv_Capsule, PrimaryCap1, PrimaryCap2, Length, ecanet_layer, AFC_layer
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.layers.merge import add
import scipy.io as scio
import cv2
import numpy as np
from PIL import Image
from random import shuffle
import os
# CBAM import *
from ST_CapsNet import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#test_path = 'E:\\code\\my_net\\bern\\data\\data7\\new_test'  # 测试集路径：样本
test_path = 'E:\\code\\SA-CapsNet\\bern\\sample\\test'  # 测试集路径：样本

def readdata(data_path):
    classes = os.listdir(data_path)
    data = []
    for cls in classes:
        files = os.listdir(data_path + "/" + cls)
        for i in files:
            img = Image.open(data_path + "/" + cls + "/" + i)
            # img = img.resize((9, 9))
            img = np.asarray(img, dtype="float32")

            data.append([img, int(cls)])
    return (data)


test_data = readdata(test_path)

y_test = [i[1] for i in test_data]
y_test = np.array(y_test)
print(y_test.shape)

def Ms_CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    st = spatial_attention(x, )

    conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', name='conv1')(st)
    conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
    conv1 = layers.Activation('relu', name='conv1_relu')(conv1)

    conv2 = layers.Conv2D(filters=16, kernel_size=3, dilation_rate=2, strides=1, padding='same', name='conv2')(conv1)
    conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
    conv2 = layers.Activation('relu', name='conv2_relu')(conv2)

    conv3 = layers.Conv2D(filters=32, kernel_size=3, dilation_rate=3, strides=1, padding='same', name='conv3')(conv2)
    conv3 = layers.BatchNormalization(momentum=0.9, name='bn3')(conv3)
    # st = spatial_attention(conv3, )

    conv3_1 = layers.Activation('relu', name='conv3_relu')(conv3)

    #  dim_vector is the dimensions of capsules, n_channels is number of feature maps
    Primary_caps1 = PrimaryCap1(conv3_1, dim_vector=8, n_channels=4, kernel_size=3, strides=2, padding='VALID')

    Conv_caps1 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps1')(Primary_caps1)

    Class_caps1 = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='class_caps1')(
        Conv_caps1)

    Class_caps_add = Class_caps1
    out_caps = Length(name='out_caps')(Class_caps_add)

    return models.Model(x, out_caps)


def cal_results(matrix):
        shape = np.shape(matrix)
        number = 0
        sum = 0
        AA = np.zeros([shape[0]], dtype=np.float)
        for i in range(shape[0]):
            number += matrix[i, i]
            AA[i] = matrix[i, i] / np.sum(matrix[i, :])
            sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
        OA = number / np.sum(matrix)
        AA_mean = np.mean(AA)
        pe = sum / (np.sum(matrix) ** 2)
        Kappa = (OA - pe) / (1 - pe)
        return OA, AA_mean, Kappa, AA


if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_class', default=2, type=int)  # number of classes
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--save_dir', default='E:\\code\\SA-CapsNet\\bern\\sample\\result_weight\\1')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)  # learning rate
    parser.add_argument('--windowsize', default=9, type=int)  # patch size
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # # define model
    model = Ms_CapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                       n_class=args.n_class,
                       num_routing=args.num_routing)
    # # model.summary()

    test_datagen = ImageDataGenerator(rescale=1. / 255)  # 不增强测试数据

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(9,9),
        batch_size=args.batch_size,
        class_mode=None, shuffle=False)


    model.load_weights('E:\\code\\SA-CapsNet\\bern\\sample\\result_weight\\1\\weights-test.h5')
    ###################################################################
    ########测试集进行预测，输出所属类别
    test_generator.reset()
    classes = model.predict_generator(test_generator,11634, verbose=0)
    #print(classes)
    pred = np.argmax(classes, axis=1)
    print(pred)
    print(pred.shape)##输出预测值

    scio.savemat('./bern/sample/result_mat/result1.mat', {"result1": pred})
########################################################################



    ####计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_test, pred)
    print(matrix)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    print('-' * 50)
    print('OA:', OA)##正确率
    print('AA:', AA_mean)
    print('Kappa:', Kappa)
    print('Classwise_acc:', AA)##每一类的正确率
