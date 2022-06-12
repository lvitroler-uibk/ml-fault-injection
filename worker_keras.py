import random
from shared.ast_parser import Visitor
from shared.utils import change_line_source
from shared.inject_functions import causeOutOfMemoryException

class WorkerKeras:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        return causeOutOfMemoryException(self.source, '.fit(', self.visitor)

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
                newSource = self.injectFoiExpandDims()

            if newSource is not None:
                break

        return newSource
    
    def injectFoiExpandDims(self):
        return None

    def injectFoiInput(self):
        return None

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        elif faultType == 'FOI':
            return self.causeFeatureInputIncompatible()
        else:
            print('Fault Type is not supported.')