import gc
import numpy
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras import optimizers
from keras import regularizers
from keras import callbacks
#from keras.utils.vis_utils import plot_model
import os
#import tensorflow as tf


# define the functions
def main():
    version = 1.2
    
    runId = 1
    passedSessions = 0
    passedLimit = 10
    errorThreshold = 4.0
    
    test_error_array= []
    last_test_error = 100.0
    best_run_index = 0
    
    #Suppress tensorflow debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #Display Banner
    displayBanner(version)
    while (passedSessions <= (passedLimit-1)):
        print('[+] Neural Network Batch Mode')
        print('[+] Running Trial ' + str(runId))
        test_error = trainingSession(str(passedSessions+1),version)
        runId = runId + 1
        if (test_error < 50.0):
            test_error_array.append(test_error)
        if test_error <= errorThreshold:
            passedSessions = passedSessions + 1
            if (test_error < last_test_error):
                last_test_error = test_error
                best_run_index = passedSessions
        print('   [-] Trial Complete. Passed=' + str(passedSessions) + ' out of ' + str(runId))
        print('   [-] Pass Rate=' + str(round(passedSessions/runId*100,2)) + '%')
        print('   [-] Best Run Index=' + str(best_run_index))
        print('   [-] Best Run Error=' + str(round(last_test_error,2))+'%')
    
    error_threshold_data = numpy.array([errorThreshold for i in range(len(test_error_array))])
    pyplot.plot(test_error_array,'r')
    pyplot.plot(error_threshold_data, 'b--')
    pyplot.xlabel('Test Session')
    pyplot.ylabel('Error (%)')
    pyplot.legend(['Test','Threshold'])
    #if (savePlotsFlag == 1):
    pyplot.savefig('testErrArray.png')
    pyplot.show()
    pyplot.close()                  
    
def trainingSession(runId,version):
    ##############################################################################    
    #configure the flags
    earlyStoppingFlag = 1
    validationOnExperimentalDataFlag = 0
    displayPlotsFlag = 0
    savePlotsFlag = 1
    saveModelFlag = 1
    normalizationFlag = 1
    dropoutInputFlag = 0
    dropoutHidden1Flag = 0
    dropoutHidden2Flag = 0
    regularizationFlag = 0
    trainingVerboseFlag = 0
    ##############################################################################    
    # configure the run parameters
    #runId = '021_'
    inputFile = 'ts20_10_1_400_acf_noiseRandom_02-acf81.txt'          # input file. Autocorrelation of a DLS ts
    #inputFile = 'ts/ts20-acf.txt'
    targetFile = 'ts20_10_1_400_d_noiseRandom_02_lsq5acr.txt'           # target file. particle diameters in nm
    testFile = 'acf_test_matlab_sm734.txt'     # test file. Autocorrelation of a DLS ts
    #testFile = 'ts/ts20-acf.txt'
    dFile = 'sm7-34-DSed.txt'         # Lorentz fit diameters in nm
    dLFile = 'sm7-34-LOR_Fit.txt'
    dAFile = 'sm7-34-ACF_Fit.txt'
    #dFile = 'ts/ts20_50_1_400_d.txt' 
    nnFile = 'nn_dls.h5'                # Neural Network save file
    reportFilename = 'run_report.txt'
    reportsDir = 'reports'
    nnmodelsDir = 'nnmodels'
    dataDir = 'ts'
    iSize = 350                         # input layer size. Should be less than max number of lags
    hSize1 = 26                          # hidden layer size
    hSize2 = 10
    oSize = 1                           # output layer size.
    pValidation = 0.15                  # percent of validation data from training data
    norm = 400                          # norm if data is normalized
    nEpochs = 40000                      # number of training epochs
    learningRate = 0.0001
    learningDecay = 0.0
    weightConstraintNorm = 999
#    kernelInWeightsInitializer = 'truncated_normal'
    kernelH1WeightsInitializer = 'truncated_normal'
    kernelH2WeightsInitializer = 'truncated_normal'
    kernelOWeightsInitializer = 'truncated_normal'
#    biasIInitializer = 'truncated_normal'
    biasH1Initializer = 'truncated_normal'
    biasH2Initializer = 'truncated_normal'
    biasOInitializer = 'truncated_normal'
#    activationI = 'linear'
    activationH1 = 'softmax'
    activationH2 = 'relu'
    activationO = 'linear'
    dropoutI = 0.2
    dropoutH1 = 0.2
    dropoutH2 = 0.2
    earlyStoppingMonitor = 'val_loss'#'val_loss' #'val_mean_absolute_percentage_error'
    optimizer = 'adam'  
    ##############################################################################    
    #Garbage collector clean memory
    gc.collect()
    if not os.path.exists(reportsDir):
        os.makedirs(reportsDir)
    if not os.path.exists(nnmodelsDir):
        os.makedirs(nnmodelsDir)
    #Suppress tensorflow debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #Init Random Seed
    numpy.random.seed()
    #Clear Norm if normalizationFlag == 0
    if (normalizationFlag == 0):
        norm = 1
    print('   [+] Starting Training Session ')#+str(runId))
    if (int(runId) < 100):
        if (int(runId) < 10):
            runId = '00' + runId
        else:
            runId = '0' + runId
    runId = runId + '_'
    ##############################################################################
    #Load training data. Returns normalized data based on the norm setting
    print('   [+] Loading training and testing data...')
    testArr, dArr, sTestSize = loadData(dataDir+'/'+testFile,dataDir+'/'+dFile,norm,iSize,oSize)
    testArr = testArr[:25,:]
    LArr, dLArr, sLorSize = loadData(dataDir+'/'+testFile,dataDir+'/'+dLFile,norm,iSize,oSize)
    AArr, dAArr, sAcfSize = loadData(dataDir+'/'+testFile,dataDir+'/'+dAFile,norm,iSize,oSize)
    inArr, tArr, sSize = loadData(dataDir+'/'+inputFile,dataDir+'/'+targetFile,norm,iSize,oSize)
    print('      [-] Complete')
    ##############################################################################
    #Create Model
    print('   [+] Building neural model...')
    model = Sequential()
    if (dropoutInputFlag == 1):
        model.add(Dropout(dropoutI, input_shape=(iSize,)))
        if (regularizationFlag == 0):
            model.add(Dense(hSize1, kernel_initializer=kernelH1WeightsInitializer,\
                            bias_initializer=biasH1Initializer,activation=activationH1,\
                            kernel_constraint=maxnorm(weightConstraintNorm)))
        else:
            model.add(Dense(hSize1, kernel_initializer=kernelH1WeightsInitializer,\
                            bias_initializer=biasH1Initializer,activation=activationH1,\
                            kernel_constraint=maxnorm(weightConstraintNorm),\
                            kernel_regularizer=regularizers.l2(0.01),\
                            activity_regularizer=regularizers.l1(0.01)))
    else:
       if (regularizationFlag == 0):
           model.add(Dense(hSize1, input_dim=iSize, kernel_initializer=kernelH1WeightsInitializer,\
                           bias_initializer=biasH1Initializer,activation=activationH1,\
                           kernel_constraint=maxnorm(weightConstraintNorm)))
       else:
           model.add(Dense(hSize1, input_dim=iSize, kernel_initializer=kernelH1WeightsInitializer,\
                           bias_initializer=biasH1Initializer,activation=activationH1,\
                           kernel_constraint=maxnorm(weightConstraintNorm),\
                           kernel_regularizer=regularizers.l2(0.01),\
                           activity_regularizer=regularizers.l1(0.01)))
    if (dropoutHidden1Flag == 1):
        model.add(Dropout(dropoutH1))
    if (hSize2 > 0):
        if (regularizationFlag == 0):
            model.add(Dense(hSize2, kernel_initializer=kernelH2WeightsInitializer, \
                            bias_initializer=biasH2Initializer, activation=activationH2,\
                            kernel_constraint=maxnorm(weightConstraintNorm)))
        else:
            model.add(Dense(hSize2, kernel_initializer=kernelH2WeightsInitializer, \
                            bias_initializer=biasH2Initializer, activation=activationH2,\
                            kernel_constraint=maxnorm(weightConstraintNorm),\
                            kernel_regularizer=regularizers.l2(0.01),\
                            activity_regularizer=regularizers.l1(0.01)))
        if (dropoutHidden2Flag == 1):
            model.add(Dropout(0.2))
        if (regularizationFlag == 0):
            model.add(Dense(oSize, kernel_initializer=kernelOWeightsInitializer,\
                            bias_initializer=biasOInitializer, activation=activationO,\
                            kernel_constraint=maxnorm(weightConstraintNorm)))
        else:
            model.add(Dense(oSize, kernel_initializer=kernelOWeightsInitializer,\
                            bias_initializer=biasOInitializer, activation=activationO,\
                            kernel_constraint=maxnorm(weightConstraintNorm),\
                            kernel_regularizer=regularizers.l2(0.01),\
                            activity_regularizer=regularizers.l1(0.01)))              
    else:
        if (dropoutHidden2Flag == 1):
            model.add(Dropout(0.2))
        if (regularizationFlag == 0):
            model.add(Dense(oSize, kernel_initializer=kernelOWeightsInitializer,\
                            bias_initializer=biasOInitializer, activation=activationO,\
                            kernel_constraint=maxnorm(weightConstraintNorm)))
        else:
            model.add(Dense(oSize, kernel_initializer=kernelOWeightsInitializer,\
                            bias_initializer=biasOInitializer, activation=activationO,\
                            kernel_constraint=maxnorm(weightConstraintNorm),\
                            kernel_regularizer=regularizers.l2(0.01),\
                            activity_regularizer=regularizers.l1(0.01)))  
    print('      [-] Complete')
    ##############################################################################
    #Compile model
    print('   [+] Compiling neural model...')
    if (optimizer == 'adam'):
        my_optimizer = optimizers.Adam(lr=learningRate, decay=learningDecay)
    elif (optimizer == 'sgd'):
        my_optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    elif (optimizer == 'rmsprop'):
        my_optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, decay=0.0)
    elif (optimizer == 'adagrad'):
        my_optimizer = optimizers.Adagrad(lr=0.01, decay=0.0)
    elif (optimizer == 'adadelta'):
        my_optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, decay=0.0)
    elif (optimizer == 'adamax'):
        my_optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.0)
    elif (optimizer == 'nadam'):
        my_optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    model.compile(loss ='mse', optimizer=my_optimizer, metrics=['mse', 'mae', 'mape'])    
    print('      [-] Complete')
    ##############################################################################
    print('   [+] Training neural model..')
    if (earlyStoppingFlag == 1):
        early_stopping = callbacks.EarlyStopping(monitor=earlyStoppingMonitor, min_delta=0.0,\
                                             patience=2, verbose=0, mode='auto')
        if (validationOnExperimentalDataFlag == 1):
            history = model.fit(inArr, tArr, validation_split=pValidation, validation_data=(testArr, dArr) ,\
                                epochs=nEpochs, batch_size=len(inArr), verbose=trainingVerboseFlag,\
                                callbacks=[early_stopping])
        else:
            history = model.fit(inArr, tArr, validation_split=pValidation,\
                                epochs=nEpochs, batch_size=len(inArr), verbose=trainingVerboseFlag,\
                                callbacks=[early_stopping])
    else:
        if (validationOnExperimentalDataFlag == 1):
            history = model.fit(inArr, tArr, validation_split=pValidation, validation_data=(testArr, dArr) ,\
                        epochs=nEpochs, batch_size=len(inArr), verbose=trainingVerboseFlag)
        else:
            history = model.fit(inArr, tArr, validation_split=pValidation,\
                        epochs=nEpochs, batch_size=len(inArr), verbose=trainingVerboseFlag)
            
    print('      [-] Complete')
    ##############################################################################
    #save the neural network
    if (saveModelFlag ==1):
        print('   [+] Saving neural model...')
        model.save(nnmodelsDir+'/'+runId+nnFile)
        print('      [-] Complete')
    ##############################################################################    
    #Plot training data
    #pyplot.plot(history.history['mean_squared_error'])
    #pyplot.xlabel('Training Epoch')
    #pyplot.ylabel('Mean Squared Error')
    #pyplot.show()
    #pyplot.plot(history.history['mean_absolute_error'])
    #pyplot.show()
    print('   [+] Generating performance plots...')
    pyplot.plot(history.history['mean_absolute_percentage_error'], 'b')
    pyplot.plot(history.history['val_mean_absolute_percentage_error'], 'r')
    pyplot.xlabel('Training Epoch')
    pyplot.ylabel('Relative Error (%)')
    pyplot.legend(['Training','Validation'])
    if (savePlotsFlag == 1):
        pyplot.savefig(reportsDir+'/'+runId+'trainingErr.png')
    if (displayPlotsFlag == 1):
        pyplot.show()
    pyplot.close()
    #Evaluate the model
    print('   [+] Evaluating Model...')
    scores = model.evaluate(inArr, tArr, verbose=0)
    print('      [-] Complete:')
    print("            %s: %.2f" % (model.metrics_names[1],scores[1]))
    print("            %s: %.2f" % (model.metrics_names[2],scores[2]))
    print("            %s: %.2f %%" % (model.metrics_names[3],scores[3]))
    #Calculate predictions for training data
    predictions_training = model.predict(inArr)
    #print(predictions_training*norm)
    pyplot.plot(tArr*norm,tArr*norm,'b--')
    pyplot.plot(tArr*norm,predictions_training*norm, 'r')
    pyplot.xlabel('Particle Diameter (nm)')
    pyplot.ylabel('Estimated Particle Diameter (nm)')
    pyplot.legend(['Ideal','Neural Prediction'])
    if (savePlotsFlag == 1):
        pyplot.savefig(reportsDir+'/'+runId+'trainingFit.png')
    if (displayPlotsFlag == 1):
        pyplot.show()
    pyplot.close()
    #Calculate predictions for test data
    predictions_testing = model.predict(testArr)
    predictions_testing = predictions_testing.reshape\
        ((predictions_testing.shape[0],1))
    ##############################################################################
    #if (validationOnExperimentalDataFlag==1):
        #Calculate sedimentation limit diameters for reference 
        #Plot predictions for test data
        #print(predictions_testing)
    tplot = numpy.arange(0, 45000, 1800)
    pyplot.plot(tplot, predictions_testing*norm, 'r')
    pyplot.plot(tplot, dLArr*norm, 'b')
    pyplot.plot(tplot, dAArr*norm, 'c')
    pyplot.plot(tplot, dArr*norm, 'g--')
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Estimated Particle Diameter (nm)')
    pyplot.legend(['Neural','Lorentz','Autocorrelation','Sedimentation'])
    if (savePlotsFlag == 1):
        pyplot.savefig(reportsDir+'/'+runId+'validationFit.png')
    if (displayPlotsFlag == 1):
        pyplot.show()
    pyplot.close()
    #else:
          #Plot predictions for test data
        #print(predictions_testing)
    #    pyplot.plot(predictions_testing*norm, 'r')
    #    pyplot.plot(dArr*norm, 'b')
    #    pyplot.xlabel('Time Series Index')
    #    pyplot.ylabel('Estimated Particle Diameter (nm)')
    #    pyplot.legend(['Neural','Lorentz'])
    #    if (savePlotsFlag == 1):
    #        pyplot.savefig(reportsDir+'/'+runId+'validationFit.png')
    #    if (displayPlotsFlag == 1):
    #        pyplot.show()
    #    pyplot.close()    
    #Calculate the relative error on test data
    deltaNNSedimentation = abs(predictions_testing - dArr)
    errRelNNSedimentation = deltaNNSedimentation/dArr*100
    deltaNNLorentz = abs(predictions_testing - dLArr)
    errRelNNLorentz = deltaNNLorentz/dLArr*100
    deltaNNAutocorrelation = abs(predictions_testing - dAArr)
    errRelNNAutocorrelation = deltaNNAutocorrelation/dAArr*100
    #if (validationOnExperimentalDataFlag==1):
    deltaL = abs(dArr - dLArr)
    deltaA = abs(dArr - dAArr)
    errRelL = deltaL/dArr*100
    errRelA = deltaA/dArr*100
    #Plot the test data results
    pyplot.plot(tplot, errRelNNSedimentation, 'r')
    pyplot.plot(tplot, errRelL, 'b')
    pyplot.plot(tplot, errRelA, 'c')
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Relative Error (%)')
    pyplot.legend(['Neural','Lorentz','Autocorrelation'])
    if (savePlotsFlag == 1):    
        pyplot.savefig(reportsDir+'/'+runId+'validationExpDataErr.png')
    if (displayPlotsFlag == 1):
        pyplot.show()
    pyplot.close()
    print('      [-] Complete')
    ##############################################################################
    nnVErr_ = min(history.history['mean_absolute_percentage_error'])
    nnTErr_ = min(history.history['val_mean_absolute_percentage_error'])
    trainCyclesCompleted = len(history.history['mean_absolute_percentage_error'])
    #print(model.summary())
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #print("=======================================================================")
    print("   [+] Train Status:")
    print("       Training Cycles Completed = %d" %trainCyclesCompleted)
    print("       Mean Best Neural Relative Error in Training = %.2f %%" % nnVErr_)
    print("       Mean Best Neural Relative Error in Testing = %.2f %%" % nnTErr_)
    #if (validationOnExperimentalDataFlag==1):
    meanErrRelNNTestSedimentation = numpy.mean(errRelNNSedimentation)
    meanErrRelNNTestLorentz = numpy.mean(errRelNNLorentz)
    meanErrRelNNTestAutocorrelation = numpy.mean(errRelNNAutocorrelation)
    meanErrRelL = numpy.mean(errRelL)
    meanErrRelA = numpy.mean(errRelA)
    print("   [+] Testing Status (Experimental Data):")
    print("       Mean Neural RelErr (Sedimentation) = %.2f %%" % meanErrRelNNTestSedimentation)
    print("       Mean Neural RelErr (Lorentz) = %.2f %%" % meanErrRelNNTestLorentz)
    print("       Mean Neural RelErr (Autocorrelation) = %.2f %%" % meanErrRelNNTestAutocorrelation)
    print("       References (Relative to Sedimentation):")
    print("         Mean Lorentz Err Rel = %.2f %%" % meanErrRelL)
    print("         Mean Autocorrelation Err Rel = %.2f %%" % meanErrRelA)
    print('   [+] Saving report...')
    saveReport(reportsDir+'/'+runId+reportFilename,version,learningRate,iSize,hSize1,hSize2,oSize,nEpochs,\
               normalizationFlag,learningDecay,sSize,norm,validationOnExperimentalDataFlag,\
               displayPlotsFlag,savePlotsFlag,dropoutInputFlag,dropoutI,dropoutHidden1Flag,\
               dropoutH1,dropoutHidden2Flag,dropoutH2,\
               regularizationFlag,earlyStoppingFlag,trainingVerboseFlag,saveModelFlag,runId+nnFile,\
               len(inArr),optimizer,weightConstraintNorm,\
               nnVErr_,nnTErr_,trainCyclesCompleted,nEpochs,earlyStoppingMonitor,runId,\
               reportsDir,nnmodelsDir,dataDir,\
               meanErrRelNNTestSedimentation,meanErrRelNNTestLorentz,meanErrRelNNTestAutocorrelation,\
               meanErrRelL, meanErrRelA)
    print('      [-] Complete')            
    ##############################################################################
    return nnTErr_

def displayBanner(version_):
    #print("\n==================================================================")
    print("[+] Dynamic Light Scattering Neural Network Builder v" + str(version_))
    #print("==================================================================")

def loadData(nIn, nTarget, tsNorm, iSize_, outSize_):    
    inAr = numpy.genfromtxt(nIn)
    inAr = inAr[:iSize_,:]
    inAr = inAr.T
    tAr = numpy.genfromtxt(nTarget)
    tAr = tAr/tsNorm
    tAr = tAr.reshape((tAr.shape[0],outSize_))
    #inAr = np.array([[0,0],[0,1],[1,0],[1,1]])
    #tAr = np.array([[0],[1],[1],[0]])
    tsSize = tAr.shape[0]
    return inAr, tAr, tsSize

def saveReport(reportFilename,v,lr,iS,hS1,hS2,oS,nE,nF,ld,tsS,nC,texpF,dPF,sPF,dIF,dI,dH1,dH2,dH1F,dH2F,rF,\
               esF,tVF,sMF,nnF,bS,optmz,wcN,nnVErr_,nnTErr_,tCC,tCL,eSM,rId,rD,mD,dD,mERNNTS,mERNNTL,mERNNTA,\
               mERL,mERA):
    buffer = "==================================================================\n"
    buffer = buffer + "Dynamic Light Scattering Neural Network v " + str(v) + "\n"
    buffer = buffer + "==================================================================\n"
    buffer = buffer + "Neural Network Parameters:\n"
    buffer = buffer + "   Input layer size (neurons)         = " + str(iS) + "\n"
    buffer = buffer + "   Hidden layer 1 size (neurons)      = " + str(hS1) + "\n"
    buffer = buffer + "   Hidden layer 2 size (neurons)      = " + str(hS2) + "\n"
    buffer = buffer + "   Output layer size (neurons)        = " + str(oS) + "\n"
    buffer = buffer + "   Number of training epochs          = " + str(nE) + "\n"
    buffer = buffer + "   Optimizer                          = " + optmz + "\n"
    buffer = buffer + "   Batch size                         = " + str(bS) + "\n"
    buffer = buffer + "   Learning rate                      = " + str(lr) + "\n"
    buffer = buffer + "   Learning decay                     = " + str(ld) + "\n"
    buffer = buffer + "   Time series size                   = " + str(tsS) + "\n"
    buffer = buffer + "   Normalization flag                 = " + str(nF) + "\n"
    buffer = buffer + "   Normalization constant             = " + str(nC) + "\n"
    buffer = buffer + "   Weight constraint norm             = " + str(wcN) + "\n"
    buffer = buffer + "   Training on experimental data flag = " + str(texpF) + "\n"
    buffer = buffer + "   Dropout on input layer             = " + str(dIF) + "\n"
    buffer = buffer + "   Dropout value input layer          = " + str(dI)+ "\n"
    buffer = buffer + "   Dropout on hidden layer 1          = " + str(dH1F) + "\n"
    buffer = buffer + "   Dropout value hidden layer 1       = " + str(dH1) + "\n"
    buffer = buffer + "   Dropout on hidden layer 2          = " + str(dH2F) + "\n"
    buffer = buffer + "   Dropout value hidden layer 2       = " + str(dH2) + "\n"
    buffer = buffer + "   Regularization                     = " + str(rF) + "\n"
    buffer = buffer + "   Early stopping                     = " + str(esF) + "\n"
    buffer = buffer + "   Early stopping monitor             = " + eSM + "\n"
    buffer = buffer + "   Training verbose                   = " + str(tVF) + "\n"
    buffer = buffer + "Execution Parameters:\n"
    buffer = buffer + "   Run id                             = " + str(rId) + "\n"
    buffer = buffer + "   Reports directory                  = " + rD + "\n"
    buffer = buffer + "   Models directory                   = " + mD + "\n"
    buffer = buffer + "   Data directory                     = " + dD + "\n"
    buffer = buffer + "   Display plots                      = " + str(dPF) + "\n"
    buffer = buffer + "   Save plot                          = " + str(sPF) + "\n"
    buffer = buffer + "   Save model                         = " + str(sMF) + "\n"
    buffer = buffer + "==================================================================" +"\n"
    buffer = buffer + "Training Results:" + "\n"
    #buffer = buffer + "   Total training duration (m:s)     = " + str(int(round(eM_,0))) +":" + str(int(round(eS_,0))) + "\n"    
    buffer = buffer + "   Training cycles completed          = " + str(tCC) + "\n"
    buffer = buffer + "   Training cycles limit              = " + str(tCL) + "\n"
    buffer = buffer + "   Network error in training          = " + str(nnVErr_) + " %\n"
    buffer = buffer + "   Network error in testing           = " + str(nnTErr_) + " %\n"
    buffer = buffer + "==================================================================" +"\n"
    buffer = buffer + "Experimental Test Results:" + "\n"
    buffer = buffer + "   Mean Neural RelErr (Sedimentation) = " + str(mERNNTS) + " %\n"
    buffer = buffer + "   Mean Neural RelErr (Lorentz) = " + str(mERNNTL) + " %\n"
    buffer = buffer + "   Mean Neural RelErr (Autocorrelation) = " + str(mERNNTA) + " %\n"
    buffer = buffer + "References (Relative to Sedimentation):" + "\n"
    buffer = buffer + "   Mean Lorentz Err Rel = " + str(mERL) + " %\n"
    buffer = buffer + "   Mean Autocorrelation Err Rel = " + str(mERA) +  "%\n"  
    buffer = buffer + "==================================================================" +"\n"
    buffer = buffer + "   Neural network saved in file:      = " + nnF +"\n"
    buffer = buffer + "==================================================================" +"\n"
    fout = open(reportFilename, 'w')
    fout.write(buffer)
    fout.close()

if __name__== "__main__":
  main()