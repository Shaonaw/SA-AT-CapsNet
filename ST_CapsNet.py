from keras import layers, models, optimizers, regularizers, constraints
from keras import backend as K
from capsulelayer_keras import Class_Capsule, Conv_Capsule, PrimaryCap1, PrimaryCap2, Length, ecanet_layer, AFC_layer
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.layers.merge import add
from keras.optimizers import SGD
import scipy.io as scio
import cv2
import numpy as np
from PIL import Image
from random import shuffle
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_path = 'E:\\code\\SA-CapsNet\\bern\\sample\\train'

def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)  #将tensor维度换位
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

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


    Class_caps_add =Class_caps1
    out_caps = Length(name='out_caps')(Class_caps_add)

    return models.Model(x, out_caps)

############################################################################
##########定义损失
def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


##############主函数
if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class', default=2, type=int)  # number of classes
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--save_dir', default='E:\\code\\SA-CapsNet\\bern\\sample\\result_weight\\1')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)  # learning rate
    parser.add_argument('--windowsize', default=9, type=int)  # patch size
    args = parser.parse_args()

    print(args)
    
    #调节权重
"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    weight_for_0 = (1 / 7100) * (11400) / 2
    weight_for_1 = (1 / 4300) * (11400) / 2

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('weight for class 0: {:.3f}'.format(weight_for_0))
    print('weight for class 1: {:.3f}'.format(weight_for_1))
"""

    # define model
    model = Ms_CapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                       n_class=args.n_class,
                       num_routing=args.num_routing)
    model.summary()
    # plot_model(model, to_file='model.png')

    # #callbacks and save the training model
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-test.h5',monitor='loss',mode='min',
                                           save_best_only=True, save_weights_only=True, verbose=1)


    # compile the model
    model.compile(optimizer=SGD(lr=args.lr, momentum=0.9),
        #optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss],
                      #loss='categorical_crossentropy',
                      #loss='binary_crossentropy',
                      metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 归一化
        horizontal_flip=True,  # 随机将一半图像水平翻转
        fill_mode='nearest'
        )

    # valid_datagen = ImageDataGenerator(rescale=1. / 255)  # 不增强验证数据


    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(9,9),
        batch_size=args.batch_size,
        class_mode='categorical',
        classes=['0', '1'],
        #shuffle=False
    )


    model.fit_generator(train_generator, epochs=args.epochs,
                        #steps_per_epoch=500,
                        #validation_data=validation_generator ,
                        #validation_steps=3,
                        class_weight=class_weight ,
                        shuffle=True, callbacks=[tb, checkpoint] , verbose=2)