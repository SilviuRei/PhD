import warnings
from keras.models import load_model
import numpy
import os
from matplotlib import pyplot

warnings.filterwarnings("ignore")

def main():
    version = 1.2
    savePlotsFlag = 1
    displayPlotsFlag = 0
    inputFile = 'acf_test_sm734.txt' 
    predictionFile = 'predictions.txt'
    idealValuesFile = 'sm7-34-DSed.txt'
    errorFile = 'error.txt'
    nnFile = 'nn_dls.h5'  
    predictionsDir = 'predictions'
    dataDir = 'ts'
    nnmodelsDir = 'nnmodels'
    reportsDir = 'reports'
    norm = 400
    iSize = 125
    maxRunId = 100
    #Suppress tensorflow debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #Display Banner
    displayBanner(version)
    if not os.path.exists(predictionsDir):
        os.makedirs(predictionsDir)
    if not os.path.exists(nnmodelsDir):
        os.makedirs(nnmodelsDir)
    print('[+] Loading Input File...')
    inputData = loadData(dataDir+'/'+inputFile, iSize)
    ideal = numpy.genfromtxt(dataDir+'/'+idealValuesFile)
    ideal = ideal.reshape(ideal.shape[0],1)
    prediction = numpy.zeros(len(ideal))
    prediction = prediction.reshape(prediction.shape[0],1)
    print('   [-] Complete')
    for runId in range(1,(maxRunId+1)):
        if (runId < 100):
            if (runId < 10):
                runIdStr = '00' + str(runId) + '_'
            else:
                runIdStr = '0' + str(runId) + '_'
        print('[+] Loading Neural Network '+str(runId)+' out of '+str(maxRunId))
        model = load_model(nnmodelsDir+'/'+runIdStr+nnFile)
        print('   [-] Complete')
        print('[+] Processing Data...')
        run_prediction = model.predict(inputData)*norm
        run_prediction = run_prediction.reshape(run_prediction.shape[0],1)
        prediction = prediction + run_prediction
        print('   [-] Complete')    
    prediction = prediction/maxRunId
    error = abs(prediction-ideal)/ideal*100
    mean_error = numpy.mean(error)
    print('[-] Network Array Mean Error = ' + str(round(mean_error,2)) + '%')
    pyplot.plot(error, 'r')
    pyplot.xlabel('Index')
    pyplot.ylabel('Relative Error (%)')
    if (savePlotsFlag == 1):    
        pyplot.savefig(reportsDir+'/'+'averagedTestErr.png')
    if (displayPlotsFlag == 1):
        pyplot.show()
    pyplot.close()
    print('[+] Saving Predictions and Errors...')
    numpy.savetxt(predictionsDir+'/'+predictionFile, prediction)
    numpy.savetxt(predictionsDir+'/'+errorFile, error)
    print('   [-] Complete')  
   
    
def loadData(fileName, iSize_):    
    inAr = numpy.genfromtxt(fileName)
    inAr = inAr[:iSize_,:]
    inAr = inAr.T
    return inAr

def displayBanner(version_):
    print("\n==================================================================")
    print("Dynamic Light Scattering Neural Network Estimator v" + str(version_))
    print("\n==================================================================")

if __name__== "__main__":
  main()