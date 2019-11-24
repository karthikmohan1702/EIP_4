 Logs for 20 epochs
 ===================
 
 Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 16s 269us/step - loss: 0.2383 - acc: 0.9246 - val_loss: 0.0944 - val_acc: 0.9685
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0690 - acc: 0.9786 - val_loss: 0.0458 - val_acc: 0.9866
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 10s 167us/step - loss: 0.0538 - acc: 0.9829 - val_loss: 0.0432 - val_acc: 0.9869
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0451 - acc: 0.9859 - val_loss: 0.0313 - val_acc: 0.9902
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0403 - acc: 0.9870 - val_loss: 0.0290 - val_acc: 0.9900
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0369 - acc: 0.9879 - val_loss: 0.0273 - val_acc: 0.9914
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0343 - acc: 0.9889 - val_loss: 0.0249 - val_acc: 0.9928
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0305 - acc: 0.9905 - val_loss: 0.0275 - val_acc: 0.9916
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0294 - acc: 0.9908 - val_loss: 0.0234 - val_acc: 0.9928
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 10s 168us/step - loss: 0.0273 - acc: 0.9910 - val_loss: 0.0207 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0251 - acc: 0.9920 - val_loss: 0.0223 - val_acc: 0.9936
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 10s 167us/step - loss: 0.0253 - acc: 0.9921 - val_loss: 0.0212 - val_acc: 0.9935
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 10s 167us/step - loss: 0.0249 - acc: 0.9917 - val_loss: 0.0196 - val_acc: 0.9945
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0230 - acc: 0.9926 - val_loss: 0.0214 - val_acc: 0.9939
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0226 - acc: 0.9927 - val_loss: 0.0213 - val_acc: 0.9941
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 10s 164us/step - loss: 0.0222 - acc: 0.9925 - val_loss: 0.0215 - val_acc: 0.9944
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0208 - acc: 0.9934 - val_loss: 0.0211 - val_acc: 0.9941
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0203 - acc: 0.9934 - val_loss: 0.0195 - val_acc: 0.9944
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0190 - acc: 0.9938 - val_loss: 0.0207 - val_acc: 0.9940
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0185 - acc: 0.9941 - val_loss: 0.0192 - val_acc: 0.9943
<keras.callbacks.History at 0x7ff8393edcc0>


*****************************************************************************************************************************


 Result of model.evaluate (on test data)
 =============================================

[0.019221754586684257, 0.9943]


*****************************************************************************************************************************


Strategy
========

-Disabled bias(using use_bias=False) & removed BatchNorm, dropout that was after the last Conv layer, as directed in the lecture. 
-used Max Pooling after the convolution of 16 filters, so we can get max (or rich) features from the convolution step.
-Then disabled dropouts & BatchNorm after the 1x1 Conv step becoz 1x1 is used to reduce the channels 
-then passed the 16 filters conv step without 1x1 to the last step so we can have rich features to the last layer.


*******************************************************************************************************************************
