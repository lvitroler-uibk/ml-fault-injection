import random
import shared.inject_functions as injections

class WorkerKeras:
    
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
        else:
            print('Fault Type is not supported.')