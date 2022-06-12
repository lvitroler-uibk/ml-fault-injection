import random
import shared.inject_functions as injections
from shared.utils import change_line_source

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
    
    def inject(self, faultType):
        if faultType == 'memory':
            return injections.causeOutOfMemoryException(self.source, 'fit', self.visitor)
        elif faultType == 'FII':
            return self.causeFeatureInputIncompatible()
        else:
            print('Fault Type is not supported.')