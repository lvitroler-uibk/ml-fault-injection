import random
import shared.inject_functions as injections
from shared.utils import change_line_source

class WorkerKeras:
    
    def __init__(self, source, visitor):
        self.source = source
        self.visitor = visitor

    def causeFeatureInputIncompatible(self):
        possibleFaults = [
            'expandDims',
            'input'
        ]
        random.shuffle(possibleFaults)
        newSource = None

        for fault in possibleFaults:
            if fault == 'input':
                newSource = self.injectFoiInput()
            elif fault == 'expandDims':
                newSource = injections.injectFoiExpandDims(self.source, 'expand_dims', self.visitor)

            if newSource is not None:
                break

        return newSource

    def injectFoiInput(self):
        return None

    def inject(self, faultType):
        if faultType == 'memory':
            return injections.causeOutOfMemoryException(self.source, '.fit(', self.visitor)
        elif faultType == 'FOI':
            return self.causeFeatureInputIncompatible()
        else:
            print('Fault Type is not supported.')