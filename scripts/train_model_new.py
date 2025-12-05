
modelName = 'Model 6' # CHANGE THIS EVERY TIME

args = {}
args['outputDir'] = 'C:/Users/jerry/OneDrive/Desktop/ECE 243A/ECE 243A Project/neural_seq_decoder/outputs/' + modelName
args['datasetPath'] = 'C:/Users/jerry/OneDrive/Desktop/ECE 243A/ECE 243A Project/neural_seq_decoder/notebooks/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32
args['lrStart'] = 0.05 # Original: 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256 # Original: 1024
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2 # Original: 0.4
args['whiteNoiseSD'] = 1.0 # Original: 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)