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
    
    def causeApiMismatch(self):
        exchanges = {
            'epochs': 'nb_epoch',
            'nb_epoch': 'epochs'
        }

        return injections.exchangeParameterNames(self.source, 'fit', exchanges, self.visitor)
    
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
        elif faultType == 'PRI':
            return injections.causeParameterRestrictionIncompatible(self.source, 'expand_dims', self.visitor)
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
        elif faultType == 'datatype':
            return injections.changeDataType(self.source, self.visitor)
        elif faultType == 'edittestdata':
            return injections.halfTestData(self.source, 'fit', self.visitor)
        elif faultType == 'layer':
            # SO 33686464
            switchFunctions = {
                'concat': 'pack',
                'pack': 'concat'
            }

            return injections.switchFunctions(self.source, switchFunctions, self.visitor)
        elif faultType == 'condition':
            return injections.addFaultyCondition(self.source, 'predict', self.visitor)
        else:
            print('Fault Type is not supported.')