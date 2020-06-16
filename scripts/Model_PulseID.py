import keras
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras.layers import Flatten, MaxPooling1D, Conv1D
from keras.models import Sequential
from keras.models import Model
from matplotlib import pyplot as plt

def get_model(filters, kernel_size, x_train, cnn_stride=1, pool_factor=1):
    #x_train = x_train.reshape((x_train.shape[0],x_train.shape[1], 1))
    #print(x_train.shape)
    #plt.imshow(x_train[0])
    inp = Input(x_train.shape[0:])
    
    convs = []
    #print(len(kernel_size))
    for k_no in range(len(kernel_size)):
        conv_size = round((x_train.shape[1] - kernel_size[k_no] + 1) / (cnn_stride)) # conv_size = number of features, output convolution signal size
        
        conv = Conv1D(filters=filters[k_no], kernel_size=kernel_size[k_no], strides=cnn_stride, padding='same',
                      dilation_rate=22, activation='relu')(inp)

        # if pool_size == strides: #we dont want overlap
        # pool_size = size of the maxpooling window
        # pool_factor = size of the output of pool 'layer'
        pool = MaxPooling1D(pool_size=int(conv_size/pool_factor), strides=abs(int(conv_size/pool_factor)), padding='valid')(conv)
        flat = Flatten()(pool)
        convs.append(flat)

    if len(kernel_size) > 1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]

    conv_model = Model(inputs=inp, outputs=out)
    model = Sequential()
    model.add(conv_model)
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=True)
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model
