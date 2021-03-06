import random
import shared.inject_functions as injections

class WorkerTensorflow:
    
    def __init__(self, source, visitor):
        self.source = source
        self.visitor = visitor

    def causeFeatureInputIncompatible(self):
        possibleFaults = [
            'expandDims',
            'modelInputShape'
        ]
        random.shuffle(possibleFaults)
        newSource = None

        for fault in possibleFaults:
            if fault == 'modelInputShape':
                newSource = injections.injectFiiModelInputShape(self.source, 'Dense', self.visitor)
            elif fault == 'expandDims':
                newSource = injections.injectFiiExpandDims(self.source, 'expand_dims', self.visitor)

            if newSource is not None:
                break

        return newSource
    
    def causeLabelOutputIncompatible(self):
        possibleFaults = [
            'modelOutputShape',
        ]
        random.shuffle(possibleFaults)
        newSource = None

        for fault in possibleFaults:
            if fault == 'modelOutputShape':
                newSource = injections.causeLoiModelOutputShape(self.source, 'Dense', self.visitor)

            if newSource is not None:
                break

        return newSource
    
    def inject(self, faultType):
        if faultType == 'memory':
            return injections.causeOutOfMemoryException(self.source, 'fit', self.visitor)
        elif faultType == 'FII':
            return self.causeFeatureInputIncompatible()
        elif faultType == 'ALI':
            return injections.causeAdjacentLayerIncompatible(self.source, 'Flatten', self.visitor)
        elif faultType == 'LOI':
            return self.causeLabelOutputIncompatible()
        elif faultType == 'API':
            return self.causeApiMismatch()
        elif faultType == 'optim':
            return injections.changeOptimiser(self.source, 'compile', self.visitor)
        elif faultType == 'hyperparams':
            return injections.worsenHyperparameters(self.source, 'fit', self.visitor)
        elif faultType == 'model':
            return injections.changeModelLoad(self.source, self.visitor)
        elif faultType == 'breakmodel':
            return injections.breakModelLoad(self.source, 'torch.load', self.visitor)
        elif faultType == 'delay':
            return injections.addDelay(self.source, 'predict', self.visitor)
        elif faultType == 'normalisation':
            return injections.removeNormalisation(self.source, 'Normalization', self.visitor)
        else:
            print('Fault Type is not supported.')