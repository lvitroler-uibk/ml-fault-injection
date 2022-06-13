import shared.inject_functions as injections
from shared.ast_parser import Visitor
from shared.utils import change_line_source

class WorkerPyTorch:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        return injections.causeOutOfMemoryException(self.source, 'DataLoader', self.visitor)

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        elif faultType == 'ALI':
            return injections.causeAdjacentLayerIncompatible(self.source, 'flatten', self.visitor)
        else:
            print('Fault Type is not supported.')