def interact_net(mode,data_folder,model_file):
    minibatchsize = 1000 # tweakable mini batch size
    import os
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    from sklearn.metrics import confusion_matrix
    # import confusion_matrix
    from tensorflow import keras
    import time
    import tensorflow

    # mode = "train"
    # data_folder = '/Users/rishabhpandey/Desktop/HW3'
    # model_file = "model.h5"
    arr = os.listdir(data_folder)
    print(len(arr))
    newarray = []
    hnn = 0
    print("Wait Processing Data")
    for i in arr:
        hnn = hnn + 1
        number = ((hnn / len(arr)) * 100)
        number = round(number, 1)
        print(str(number) + "%",end="\r")
        val = []
        img = Image.open(data_folder + '/' + i)
        im_matrix = np.array(img)

        name = i.split(".")[0]
        val.append(name)
        k = -1
        for d in im_matrix:
            k = k + 1

            lt = -1
            for s in im_matrix[k]:
                lt = lt + 1
                value = 0
                for v in im_matrix[k][lt]:
                    value = value + (v / 255)
                newval = value / 3
                val.append(newval)
        newarray.append(val)

    k = ["label"]
    for i in range(1, 4097):
        k.append("pixel" + str(i))

    import pandas as pd

    df = pd.DataFrame(newarray, columns=k)

    g = -1
    for lt in df['label']:
        g = g + 1
        if lt.find("NO_COLLISIO") != -1:
            df['label'][g] = 0
        elif lt.find("_INELASTIC_DECAY") != -1:
            df['label'][g] = 3
        elif lt.find("_INELASTIC") != -1:
            df['label'][g] = 1
        elif lt.find("_ELASTIC") != -1:
            df['label'][g] = 2

    n = 10
    lt = df.head(int(len(df) * (n / 100)))


    def percentage(part, whole):
        return 100 * float(part) / float(whole)


    new = percentage(len(lt), len(df))
    df = df.iloc[len(lt):]

    df.to_csv('particletrain.csv', index=False)
    lt.to_csv('particletest.csv', index=False)

    import pandas as pd

    train_data = pd.read_csv("particletrain.csv")
    test_data = pd.read_csv("particletest.csv")

    Y_train = train_data["label"]  # defining labels as Y_train
    Y_test = test_data["label"]
    X_train = train_data.drop(labels=["label"], axis=1)  # defining the images as X_train
    X_test = test_data.drop(labels=["label"], axis=1)
    # g = plt.imshow(X_train[100])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.values.reshape(X_train.shape[0], 64, 64, 1)
    X_test = X_test.values.reshape(X_test.shape[0], 64, 64, 1)

    try:
        from keras.utils.np_utils import to_categorical
    except:
        from tensorflow.keras.utils import to_categorical
    Y_train = to_categorical(Y_train, num_classes=4)
    Y_test = to_categorical(Y_test, num_classes=4)

    if mode == "test":
        import keras
        from keras.models import Sequential
        from keras.datasets import mnist
        from keras.layers import Dense
        from tensorflow.keras.optimizers import SGD
        # from keras.optimizers import Adam
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
        import tensorflow

        model1 = tensorflow.keras.models.load_model(model_file)
        # model1.predict(X_test)
        start = time.time()
        _, acc = model1.evaluate(X_test, Y_test, verbose=0)
        print()
        print("Total Accuracy = ", '> %.3f' % (acc * 100.0), "%")
        print()
        import numpy as np

        y_pred = model1.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(Y_test, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix")
        print(cm)
        print()
        end = time.time()
        print(f"Time taken to Test Model Accuracy {end - start}")
        print()

    if mode == "train":
        import keras
        from keras.models import Sequential
        from keras.datasets import mnist
        from keras.layers import Dense
        from tensorflow.keras.optimizers import SGD
        # from keras.optimizers import Adam
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='softmax'))
        # compile model
        optimizer = SGD(learning_rate=0.01, momentum=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        start = time.time()
        try:
            Numbertrain = len(X_train)
            Numbertest = len(X_test)
        except:
            pass
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=minibatchsize, verbose=1, shuffle=1)
        end = time.time()
        print(f"Time taken to train Model {end - start}")
        start = time.time()
        _, acc = model.evaluate(X_test, Y_test, verbose=0)
        print()
        print("Total Accuracy = ", '> %.3f' % (acc * 100.0))
        print()
        import numpy as np

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(Y_test, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix")
        print(cm)
        print()
        end = time.time()
        print(f"Time taken to Test Model Accuracy {end - start}")
        print()
        try:
            model.save(model_file)
        except:
            model.save(model_file + '.h5')
        print("Number of Training Data", Numbertrain)
        print("Number of Testing Data", Numbertest)
        print("processing complete")

    if mode == "5fold":
        import keras
        from keras.models import Sequential
        from keras.datasets import mnist
        from keras.layers import Dense
        from tensorflow.keras.optimizers import SGD
        # from keras.optimizers import Adam
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

        from numpy import mean
        from numpy import std
        from sklearn.model_selection import KFold
        from keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        from keras.models import Sequential
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import Dense
        from keras.layers import Flatten
        from sklearn.metrics import confusion_matrix
        from tensorflow.keras.optimizers import SGD


        # load train and test dataset
        def load_dataset():
            trainY = train_data["label"]  # defining labels as Y_train
            trainX = train_data.drop(labels=["label"], axis=1)  # defining the images as X_train
            testX = test_data.drop(labels=["label"], axis=1)
            testY = test_data["label"]
            trainX = trainX.values.reshape(trainX.shape[0], 64, 64, 1)
            testX = testX.values.reshape(testX.shape[0], 64, 64, 1)
            trainY = to_categorical(trainY)
            testY = to_categorical(testY)
            return trainX, trainY, testX, testY


        # scale pixels
        def prep_pixels(train, test):
            # convert from integers to floats
            train_norm = train.astype('float32')
            test_norm = test.astype('float32')
            # normalize to range 0-1
            # train_norm = train_norm / 255.0
            # test_norm = test_norm / 255.0
            # return normalized images
            return train_norm, test_norm


        # define cnn model
        def define_model():
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 1)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(4, activation='softmax'))
            # compile model
            opt = SGD(lr=0.01, momentum=0.0)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            return model


        # evaluate a model using k-fold cross-validation
        def evaluate_model(dataX, dataY, n_folds=5):
            scores, histories = list(), list()
            # prepare cross validation
            kfold = KFold(n_folds, shuffle=True, random_state=1)
            # enumerate splits
            rr = 0
            for train_ix, test_ix in kfold.split(dataX):
                rr = rr + 1
                # define model
                model = define_model()
                # select rows for train and test
                trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
                # fit model
                try:
                    Numbertrain = len(X_train)
                    Numbertest = len(X_test)
                except:
                    Numbertrain = ""
                    Numbertest = ""
                start = time.time()
                history = model.fit(trainX, trainY, epochs=10, batch_size=minibatchsize, validation_data=(testX, testY), verbose=1)
                end = time.time()
                print(f"Time taken to train Model {end - start}")
                # evaluate model
                start = time.time()
                _, acc = model.evaluate(testX, testY, verbose=0)
                print()
                print("Total Accuracy = ", '> %.3f' % (acc * 100.0))
                print()
                import numpy as np
                y_pred = model.predict(testX)
                y_pred = np.argmax(y_pred, axis=1)
                y_test = np.argmax(testY, axis=1)
                cm = confusion_matrix(y_test, y_pred)
                print("Confusion Matrix")
                print(cm)
                print()
                end = time.time()
                print(f"Time taken to Test Model Accuracy {end - start}")
                print()
                print("Number of Training Data", Numbertrain)
                print("Number of Testing Data", Numbertest)

                # stores scores
                scores.append(acc)
                histories.append(history)
                if rr == 5:
                    try:
                        model.save(model_file)
                    except:
                        model.save(model_file + '.h5')
            try:
                print("processing complete")
            except:
                pass
            return scores, histories


        # run the test harness for evaluating a model
        def run_test_harness():

            trainX, trainY, testX, testY = load_dataset()

            trainX, testX = prep_pixels(trainX, testX)

            scores, histories = evaluate_model(trainX, trainY)


        run_test_harness()

import sys
i = 0
for arg in sys.argv:

	globals()["arg" + str(i)] = arg
	i = i+1

mode = arg1
data_folder = arg2
model_file = arg3

interact_net(mode,data_folder,model_file)

