from shared.ast_parser import Visitor
from shared.utils import change_line_source
from shared.inject_functions import causeOutOfMemoryException

class WorkerPyTorch:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        return causeOutOfMemoryException(self.source, 'DataLoader(', self.visitor)

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        else:
            print('Fault Type is not supported.')