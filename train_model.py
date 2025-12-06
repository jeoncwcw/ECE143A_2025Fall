#Chris
modelName = 'speechBaseline4'

args = {}
args['outputDir'] = f'/mnt/data/project/neural_seq_decoder/logs/speech_logs/{modelName}'
args['datasetPath'] = '/mnt/data/project_data/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5


args['augmentTimeMask'] = False # True= enabled
args['maxMaskLen'] = 20
args['timeMaskProb'] = 0.5

args['augmentFeatureMask'] = False # True= enabled
args['featureMaskProb'] = 0.5
args['maxFeatureMask'] = 10



from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)