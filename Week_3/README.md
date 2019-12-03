Final Validation accuracy for Base Network
===========================================

Accuracy on test data is: 82.67

Model definition (model.add... ) with output channel size and receptive field
=============================================================================

model = Sequential()
model.add(SeparableConv2D(filters = 64, kernel_size=(3, 3),padding = 'valid', strides = (1,1),
                          activation='relu',dilation_rate=1,depth_multiplier = 1,input_shape = (32,32,3)))
model.add(BatchNormalization())
model.add(Dropout(0.1)) #In - 32, Out - 30, Rout - 3, Jout - 1 

model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
model.add(BatchNormalization()) #In - 30, Out - 28, Rout - 5, Jout - 1 

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) #In - 28, Out - 14, Rout - 6, Jout - 2 

model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
model.add(BatchNormalization()) #In - 14, Out - 12, Rout - 10, Jout - 2 

model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
model.add(BatchNormalization())
model.add(Dropout(0.1)) #In - 12, Out - 10, Rout - 14, Jout - 2 

model.add(SeparableConv2D(filters = 64, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
model.add(BatchNormalization()) #In - 10, Out - 8,  Rout - 18, Jout - 2 

model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
model.add(BatchNormalization()) #In - 8,  Out - 6,  Rout - 22, Jout - 2 

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) #In - 6,  Out - 3,  Rout - 24, Jout - 4 

model.add(SeparableConv2D(filters = 10, kernel_size=(3, 3),padding = 'valid', strides = (1,1),activation='relu',depth_multiplier = 1))
#In - 3,  Out - 1,  Rout - 26, Jout - 4 

model.add(Flatten())

model.add(Activation('softmax'))
# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])




50 epoch logs
==============
Epoch 1/50
/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '
/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:724: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '
390/390 [==============================] - 56s 145ms/step - loss: 1.6767 - acc: 0.3908 - val_loss: 1.5357 - val_acc: 0.4983
Epoch 2/50
390/390 [==============================] - 48s 123ms/step - loss: 1.3134 - acc: 0.5282 - val_loss: 1.0994 - val_acc: 0.6215
Epoch 3/50
390/390 [==============================] - 48s 123ms/step - loss: 1.1858 - acc: 0.5784 - val_loss: 1.2905 - val_acc: 0.5930
Epoch 4/50
390/390 [==============================] - 48s 123ms/step - loss: 1.0999 - acc: 0.6099 - val_loss: 1.0368 - val_acc: 0.6506
Epoch 5/50
390/390 [==============================] - 48s 123ms/step - loss: 1.0382 - acc: 0.6349 - val_loss: 0.9909 - val_acc: 0.6603
Epoch 6/50
390/390 [==============================] - 48s 123ms/step - loss: 0.9868 - acc: 0.6523 - val_loss: 0.8124 - val_acc: 0.7223
Epoch 7/50
390/390 [==============================] - 47s 122ms/step - loss: 0.9463 - acc: 0.6678 - val_loss: 0.8433 - val_acc: 0.7172
Epoch 8/50
390/390 [==============================] - 47s 121ms/step - loss: 0.9167 - acc: 0.6807 - val_loss: 0.8871 - val_acc: 0.7031
Epoch 9/50
390/390 [==============================] - 47s 122ms/step - loss: 0.8895 - acc: 0.6881 - val_loss: 0.8402 - val_acc: 0.7230
Epoch 10/50
390/390 [==============================] - 47s 121ms/step - loss: 0.8672 - acc: 0.6977 - val_loss: 0.7980 - val_acc: 0.7297
Epoch 11/50
390/390 [==============================] - 47s 122ms/step - loss: 0.8440 - acc: 0.7075 - val_loss: 0.8049 - val_acc: 0.7412
Epoch 12/50
390/390 [==============================] - 48s 122ms/step - loss: 0.8259 - acc: 0.7131 - val_loss: 0.7258 - val_acc: 0.7502
Epoch 13/50
390/390 [==============================] - 48s 122ms/step - loss: 0.8164 - acc: 0.7174 - val_loss: 0.7534 - val_acc: 0.7528
Epoch 14/50
390/390 [==============================] - 48s 122ms/step - loss: 0.7964 - acc: 0.7222 - val_loss: 0.7227 - val_acc: 0.7577
Epoch 15/50
390/390 [==============================] - 48s 122ms/step - loss: 0.7728 - acc: 0.7317 - val_loss: 0.7232 - val_acc: 0.7610
Epoch 16/50
390/390 [==============================] - 47s 122ms/step - loss: 0.7695 - acc: 0.7338 - val_loss: 0.7187 - val_acc: 0.7558
Epoch 17/50
390/390 [==============================] - 47s 121ms/step - loss: 0.7630 - acc: 0.7347 - val_loss: 0.6809 - val_acc: 0.7709
Epoch 18/50
390/390 [==============================] - 47s 121ms/step - loss: 0.7493 - acc: 0.7417 - val_loss: 0.6869 - val_acc: 0.7662
Epoch 19/50
390/390 [==============================] - 47s 121ms/step - loss: 0.7405 - acc: 0.7441 - val_loss: 0.5953 - val_acc: 0.7954
Epoch 20/50
390/390 [==============================] - 47s 120ms/step - loss: 0.7345 - acc: 0.7451 - val_loss: 0.7020 - val_acc: 0.7708
Epoch 21/50
390/390 [==============================] - 47s 119ms/step - loss: 0.7167 - acc: 0.7502 - val_loss: 0.6084 - val_acc: 0.7902
Epoch 22/50
390/390 [==============================] - 47s 120ms/step - loss: 0.7146 - acc: 0.7519 - val_loss: 0.6666 - val_acc: 0.7800
Epoch 23/50
390/390 [==============================] - 46s 119ms/step - loss: 0.7034 - acc: 0.7566 - val_loss: 0.6346 - val_acc: 0.7856
Epoch 24/50
390/390 [==============================] - 47s 120ms/step - loss: 0.6983 - acc: 0.7581 - val_loss: 0.6061 - val_acc: 0.7926
Epoch 25/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6903 - acc: 0.7602 - val_loss: 0.6404 - val_acc: 0.7887
Epoch 26/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6848 - acc: 0.7625 - val_loss: 0.6046 - val_acc: 0.7983
Epoch 27/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6761 - acc: 0.7653 - val_loss: 0.6605 - val_acc: 0.7802
Epoch 28/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6723 - acc: 0.7680 - val_loss: 0.6445 - val_acc: 0.7856
Epoch 29/50
390/390 [==============================] - 47s 122ms/step - loss: 0.6655 - acc: 0.7686 - val_loss: 0.6136 - val_acc: 0.7975
Epoch 30/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6629 - acc: 0.7700 - val_loss: 0.6055 - val_acc: 0.8001
Epoch 31/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6525 - acc: 0.7742 - val_loss: 0.6472 - val_acc: 0.7821
Epoch 32/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6523 - acc: 0.7735 - val_loss: 0.6752 - val_acc: 0.7830
Epoch 33/50
390/390 [==============================] - 47s 122ms/step - loss: 0.6444 - acc: 0.7777 - val_loss: 0.6258 - val_acc: 0.7969
Epoch 34/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6411 - acc: 0.7776 - val_loss: 0.6140 - val_acc: 0.7925
Epoch 35/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6304 - acc: 0.7820 - val_loss: 0.6187 - val_acc: 0.8009
Epoch 36/50
390/390 [==============================] - 47s 121ms/step - loss: 0.6343 - acc: 0.7796 - val_loss: 0.5891 - val_acc: 0.8015
Epoch 37/50
390/390 [==============================] - 47s 121ms/step - loss: 0.6298 - acc: 0.7832 - val_loss: 0.5617 - val_acc: 0.8099
Epoch 38/50
390/390 [==============================] - 47s 121ms/step - loss: 0.6261 - acc: 0.7841 - val_loss: 0.5673 - val_acc: 0.8087
Epoch 39/50
390/390 [==============================] - 47s 121ms/step - loss: 0.6184 - acc: 0.7862 - val_loss: 0.5812 - val_acc: 0.8084
Epoch 40/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6130 - acc: 0.7896 - val_loss: 0.6335 - val_acc: 0.7953
Epoch 41/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6117 - acc: 0.7874 - val_loss: 0.5622 - val_acc: 0.8110
Epoch 42/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6071 - acc: 0.7901 - val_loss: 0.7199 - val_acc: 0.7726
Epoch 43/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6029 - acc: 0.7904 - val_loss: 0.5567 - val_acc: 0.8125
Epoch 44/50
390/390 [==============================] - 48s 122ms/step - loss: 0.6046 - acc: 0.7913 - val_loss: 0.5863 - val_acc: 0.8076
Epoch 45/50
390/390 [==============================] - 47s 122ms/step - loss: 0.5945 - acc: 0.7947 - val_loss: 0.5768 - val_acc: 0.8140
Epoch 46/50
390/390 [==============================] - 48s 122ms/step - loss: 0.5941 - acc: 0.7949 - val_loss: 0.5616 - val_acc: 0.8158
Epoch 47/50
390/390 [==============================] - 47s 122ms/step - loss: 0.5980 - acc: 0.7945 - val_loss: 0.5664 - val_acc: 0.8157
Epoch 48/50
390/390 [==============================] - 48s 122ms/step - loss: 0.5874 - acc: 0.7957 - val_loss: 0.6116 - val_acc: 0.8037
Epoch 49/50
390/390 [==============================] - 47s 122ms/step - loss: 0.5842 - acc: 0.7995 - val_loss: 0.5552 - val_acc: 0.8156
Epoch 50/50
390/390 [==============================] - 47s 122ms/step - loss: 0.5864 - acc: 0.7973 - val_loss: 0.5330 - val_acc: 0.8209
Model took 2382.86 seconds to train

Accuracy on test data is: 82.09



