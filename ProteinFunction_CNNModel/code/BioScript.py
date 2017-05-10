    # Import
# Need to check that the following modules plus h5py in installed. How?
import math
import gc
import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from Bio.Alphabet import generic_protein
from keras.models import Sequential
from keras import callbacks
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef

# Global variables
input_dataPath = "./data/"
output_verbose = True
#maxSeqLen = 4 # This denotes how many letters at a time each
protSeqConcat = np.zeros(0)
sampleCount = 10
sampleLen = 10
totalwidth=sampleCount*sampleLen*21  # 21 = one hot vecter length - see furhter below
loadFromSave = False

shuffleCount = 1  #3
crossVal=3  # must be 2 or greater
numOfEpochs=5  # 15
model_learningRate = 0.01  # 0.01

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

class ProteinSequencingCNNModel:
    def __init__(self):
        self.input_Height = 1   # Set to 0 to start with
        self.input_Width =1   # Set to 0 to start with
        self.input_Channels = 1  #(e.g. 3 for RGB)
        self._input_TrainData=[]
        self._input_TrainLabels = []
        self._input_TestData = []
        self._input_PredData = []
        self._input_TestLabels = []

        self.output_FilePath =""
        self.output_Filename="ProteinSequencingCNNModel.h5"
        self.model_batchsize = 64
        self.model_epochs=1
        self.model = Sequential()

    def loadTrainData(self,trainData,trainLabels):

        #self._input_TrainData = trainData.reshape(trainData.shape[0], 1, trainData.shape[1], 1)  # n, depth, width, height: 8722,1,630,1
        self._input_TrainData = trainData.reshape(trainData.shape[0], 1, trainData.shape[1],1)

        print(trainData.shape)
        print(self._input_TrainData.shape)

        self._input_TrainLabels = trainLabels

    def loadTestData(self,testData,testLabels=[]):
        self._input_TestData = testData.reshape(testData.shape[0], 1, testData.shape[1], 1)  # n, depth, width, height: 8722,1,630,1
        self._input_TestLabels = testLabels

    def saveModel(self):
        if self.output_FilePath == "": self.output_FilePath=os.getcwd() + os.path.sep + self.output_Filename
        self.model.save(self.output_FilePath)
        print('Model saved to:' + self.output_FilePath)

    def loadModel(self):
        if self.output_FilePath == "": self.output_FilePath=os.getcwd() + os.path.sep + self.output_Filename
        self.model =load_model(self.output_FilePath)
        print('Model loaded:' + self.output_FilePath)

    def createModel(self):
        # https://keras.io/getting-started/sequential-model-guide/
        # Reset model

        model_params=[1,32,1,1]

        self.model = Sequential()

        # Layer 1
        #From TF Conv2DFunction: input_shape = (128, 128, 3) for 128x128 RGB pictures in `data_format where "channels_last"

        self.model.add(Conv2D(32, (1, 3), activation='relu', input_shape=(self.input_Height, self.input_Width, self.input_Channels)))  #
        self.model.add(Conv2D(32, (1, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))
        self.model.add(Dropout(0.25))

        # Layer 2
        self.model.add(Conv2D(64, (1, 1), activation='relu'))
        self.model.add(Conv2D(64, (1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 1)))
        self.model.add(Dropout(0.25))

        # Layer 3
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, activation='softmax'))

        # Layer 4
        sgd = SGD(lr=model_learningRate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return self.model

    def initializeShapes(self):
        # Any code that should be run just prior to running the model
        self.input_Width = totalwidth
        self.createModel()

    def predict(self):
        return self.model.predict(self._input_TestData)

    def evaluate(self):
        gc.collect()  # Bugfix: See https://github.com/tensorflow/tensorflow/issues/3388  -- seems to work without for now
        return self.model.evaluate (self._input_TestData, self._input_TestLabels, self.model_batchsize)

    def fit(self):
        self.model.fit(self._input_TrainData, self._input_TrainLabels, batch_size=self.model_batchsize,epochs=self.model_epochs)
        gc.collect()  # Bugfix: See https://github.com/tensorflow/tensorflow/issues/3388  -- seems to work without for now

def shuffleTwoArrays(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def manualOneHot(inpStr):
    _res=""
    # 21 char one-hot encoder
    for inp in inpStr:
        l=len(_res)
        if (inp == 'Q'):  _res+= '100000000000000000000'
        if (inp == 'S'):  _res+= '010000000000000000000'
        if (inp == 'R'):  _res+= '001000000000000000000'
        if (inp == 'F'):  _res+= '000100000000000000000'
        if (inp == 'P'):  _res+= '000010000000000000000'
        if (inp == 'K'):  _res+= '000001000000000000000'
        if (inp == 'I'):  _res+= '000000100000000000000'
        if (inp == 'N'):  _res+= '000000010000000000000'
        if (inp == 'H'):  _res+= '000000001000000000000'
        if (inp == 'C'):  _res+= '000000000100000000000'
        if (inp == 'W'):  _res+= '000000000010000000000'
        if (inp == 'M'):  _res+= '000000000001000000000'
        if (inp == 'Y'):  _res+= '000000000000100000000'
        if (inp == 'A'):  _res+= '000000000000010000000'
        if (inp == 'V'):  _res+= '000000000000001000000'
        if (inp == 'E'):  _res+= '000000000000000100000'
        if (inp == 'G'):  _res+= '000000000000000010000'
        if (inp == 'L'):  _res+= '000000000000000001000'
        if (inp == 'T'):  _res+= '000000000000000000100'
        if (inp == 'D'):  _res+= '000000000000000000010'
        if (inp == 'U'):  _res+= '000000000000000000001'
        if (inp == 'X'):  _res+= '111111111111111111111'
        if (inp == 'Z'):  _res+= '000000000000000000000'  # Padding char
        if l==len(_res):
            print('Unrecognized letter:'+ inp)
    return _res

def getStats(_seq):
    _avLen=0
    _maxLen=0
    count =0
    for s in _seq:
        _avLen+=len(s)
        count+=1
        _maxLen=max(len(s),_maxLen)

    return _maxLen,_avLen/count

def preprocess(_seq: SeqIO,classLabel):
    # Go through each sequence and randomly select samples from sequence of fixed length
    tmp=[]
    _classLabels=[]
    totalSequences=0
    tmpS=[]
    seqID = []

    for s in _seq:  # Go through every sequence - e.g. CGAGAGACSBAAGGA
        res = ''
        totalSequences+=1

        seqStr = str(s.seq)
        gap=(len(seqStr)-sampleCount*sampleLen)/sampleCount  # Calculate the gap to leave in between sampling from sequence
        if gap<0:
            # Sequence is too short
            res=seqStr+('Z'*(sampleCount*sampleLen-len(seqStr)))
        else:
            pos=0
            for i in range(0,sampleCount):
                res+=seqStr[pos:pos+sampleLen]
                pos+=math.floor(gap)


        if(len(res)!=sampleCount*sampleLen):  # Check id
            print ('Err'+ str(len(res )))
            exit()
        tmp.append(manualOneHot(res))
        _classLabels.append(classLabel)
        seqID.append(s.id)

    tmp = np.array(tmp)

    # Declare an empty array to hold the final results
    _oneHotSamples = np.zeros((len(tmp), sampleLen*sampleCount*21))  # 21 = onehot len
    for i,s in enumerate(tmp):
        # iterate through the tmp
        #_oneHotSamples[i] = list(s).append(len(s))      # Add on the length of the sequence at the end
        #_oneHotSamples[i] = list(s).append(0)  # Add on the length of the sequence at the end
        _oneHotSamples[i] = list(s)
    _classLabels=np.reshape(_classLabels,(len(_classLabels),4))

    return _oneHotSamples, _classLabels,seqID


################################## START OF MAIN CODE ###############################

# Load data and preprocess
cyto= SeqIO.parse(input_dataPath + "cyto.fasta", "fasta",generic_protein)
oneHotSamples,classLabels,_=preprocess(cyto,np.array([1,0,0,0])) # 1 for cyto

mito= SeqIO.parse(input_dataPath + "mito.fasta", "fasta",generic_protein)
s,c,_=preprocess(mito,np.array([0,1,0,0])) # 1 for cyto
oneHotSamples= np.concatenate((oneHotSamples,s),axis=0)
classLabels =np.concatenate((classLabels,c),axis=0)

nucleus= SeqIO.parse(input_dataPath + "nucleus.fasta", "fasta",generic_protein)
s,c,_=preprocess(nucleus,np.array([0,0,1,0])) # 1 for cyto
oneHotSamples= np.concatenate((oneHotSamples,s),axis=0)
classLabels =np.concatenate((classLabels,c),axis=0)


secreted= SeqIO.parse(input_dataPath + "secreted.fasta", "fasta",generic_protein)
s,c,_=preprocess(secreted,np.array([0,0,0,1])) # 1 for cyto
oneHotSamples= np.concatenate((oneHotSamples,s),axis=0)
classLabels =np.concatenate((classLabels,c),axis=0)

myModel=ProteinSequencingCNNModel()         # Create the model object
# To load model:
if loadFromSave:
    myModel.loadModel()
else:
    myModel.initializeShapes()  # Get the object to automatically infer shapes


# Shuffle data
shuffledSamples,shuffledLabels = shuffleTwoArrays(oneHotSamples,classLabels)

# Retain last 100 for validation purposes
validationSamples = shuffledSamples[-100:]
validationLabels=shuffledLabels[-100:]
shuffledSamples=shuffledSamples[0:-100]
shuffledLabels=shuffledLabels[0:-100]

for i in range(0,shuffleCount):

    # Shuffle data again
    shuffledSamples, shuffledLabels = shuffleTwoArrays(shuffledSamples, shuffledLabels)

    # K folds cross val
    kf = KFold(n_splits=crossVal)
    j=0
    # Do cross val on shuffled
    for train_index, test_index in kf.split(shuffledSamples):
        j+=1
        # Set up train and test data
        l=len(shuffledSamples)

        x_train, x_test = shuffledSamples[train_index], shuffledSamples[test_index]
        y_train, y_test = shuffledLabels[train_index], shuffledLabels[test_index]

        # ------------------Keras section ------------------
        # In input_shape the batch dimension is not specified   https://keras.io/getting-started/sequential-model-guide/
        # If your image batch is of N images of HxW size with C channels, theano uses the NCHW ordering while tensorflow uses the NHWC ordering.

        myModel.model_epochs=numOfEpochs
        myModel.loadTrainData(x_train, y_train)  # Load the training data and convert it to the right shape
        myModel.fit()                               # Run the model
        gc.collect()  # Bugfix: See https://github.com/tensorflow/tensorflow/issues/3388  -- seems to work without for now
        myModel.saveModel()                         # Save model
        print('Cross-val: ',j)

    # To test model:
    myModel.loadTestData(x_test,y_test)         # Load test data and convert it to the right shape
    gc.collect()  # Bugfix: See https://github.com/tensorflow/tensorflow/issues/3388  -- seems to work without for now
    score = myModel.evaluate()
    print('\nShuffle count' + str(i) + ' of ' + str(shuffleCount) + ' Test score:' + str(score))


####### Validation
myModel.loadTestData(validationSamples,validationLabels) # Load validation data and convert it to the right shape
gc.collect()
score = myModel.evaluate()
prob = myModel.predict()
gc.collect()
pred=np.argmax(prob,axis=1)
gc.collect()
print('\nFinished, Validation score:' + str(score))
## Combine into one panda
df = pd.DataFrame(validationLabels)

df = df.assign(prob_Cyto=pd.Series(prob.T[0]))
df = df.assign(prob_Mito=pd.Series(prob.T[1]))
df = df.assign(prob_Nuc=pd.Series(prob.T[2]))
df = df.assign(prob3_Sec=pd.Series(prob.T[3]))
df = df.assign(pred=pd.Series(pred))
df.to_csv('Validation.csv')

###### Final submission
evalData= SeqIO.parse(input_dataPath + "blind.fasta", "fasta",generic_protein)
oneHotSamples,oneHotLabels,sampleIDs = preprocess(evalData,np.array([0,0,0,0]))
myModel.loadTestData(oneHotSamples)
print('\n')
prob = myModel.predict()
gc.collect()
pred=np.argmax(prob,axis=1)
gc.collect()

## Combine into one panda
df = pd.DataFrame(sampleIDs)
df = df.assign(prob_Cyto=pd.Series(prob.T[0]))
df = df.assign(prob_Mito=pd.Series(prob.T[1]))
df = df.assign(prob_Nuc=pd.Series(prob.T[2]))
df = df.assign(prob3_Sec=pd.Series(prob.T[3]))
df = df.assign(pred=pd.Series(pred))
df.to_csv('Submission.csv')