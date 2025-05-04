from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import librosa
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from playsound import playsound
import speech_recognition as speechRecog

main = tkinter.Tk()
main.title("CNN BASED SPEAKER RECOGNITION IN LANGUAGE AND TEXT-INDEPENDENT SMALL-SCALE SYSTEM")
main.geometry("1200x1200")

global classifier
global X, Y

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

def mfccProcessing():
    global X, Y
    '''
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            x, sr = librosa.load(root+"/"+directory[j])
            mfccs = librosa.feature.mfcc(x, sr=sr)
            temp = np.reshape(mfccs,(mfccs.shape[0],mfccs.shape[1],1))
            X_train.append(temp)
            Y_train.append(getID(name))
            print(name+" "+root+"/"+directory[j]+" "+str(temp.shape))
    '''
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"Total speech audio files found in dataset is : "+str(len(X))+" from "+str(Y.shape[1])+" different persons\n")
    

def runCNN():
    global classifier
    text.delete('1.0', END)
    global X, Y
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Speech Recognition Training Model Prediction Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 5, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Speech Recognition Training Model Prediction Accuracy = "+str(accuracy))
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(accuracy, 'ro-', color = 'orange')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['CNN Accuracy', 'CNN Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('CNN Accuracy & Loss Comparison Graph')
    plt.show()


def runRandomForest():
    global X, Y
    XX = np.reshape(X,(X.shape[0],(X.shape[1] * X.shape[2] * X.shape[3])))
    YY = Y.argmax(axis=1)
    pca = PCA(n_components = 80)
    XX = pca.fit_transform(XX)
    print(XX.shape)
    print(YY.shape)
    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test) 
    random_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,'\nRandom Forest Prediction Accuracy : '+str(random_acc)+"\n")

def predict():
    text.delete('1.0', END)
    names = ['Benjamin Netanyau','Jens Stoltenberg','Magaret Tarcher','Julia Gillard','Nelson Mandela']
    filename = filedialog.askopenfilename(initialdir="testSpeech")
    x, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    test = np.reshape(mfccs,(mfccs.shape[0],mfccs.shape[1],1))
    test = test.reshape(1,20,44,1)
    preds = classifier.predict(test)
    predict = np.argmax(preds)
    person = names[predict]
    text.insert(END,'Uploaded Speech Recognized for Person : '+str(person)+"\n\n")
    rr = speechRecog.Recognizer()
    with speechRecog.WavFile(filename) as sourceAudio:
        audioData = rr.record(sourceAudio)                        
    try:
        text.insert(END,"Voice Recognization as : " + rr.recognize_google(audioData)+"\n\n")
    except LookupError:                                 
        text.insert(END,"Could not understand audio")
    playsound(filename)
    
    

font = ('times', 15, 'bold')
title = Label(main, text='CNN BASED SPEAKER RECOGNITION IN LANGUAGE AND TEXT-INDEPENDENT SMALL-SCALE SYSTEM')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Speech Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=380,y=100)

noiseButton = Button(main, text="Noise & MFCC Processing", command=mfccProcessing)
noiseButton.place(x=50,y=150)
noiseButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithms", command=runRandomForest)
rfButton.place(x=50,y=250)
rfButton.config(font=font1)

graphButton = Button(main, text="CNN Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Upload Test Audio & Recognize Speaker", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
