from shared.utils import change_line_source
import random
import shared.inject_functions as injections
from shared.ast_parser import Visitor

class WorkerScitkit:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def inject(self, faultType):
        if faultType == 'memory':
            return injections.causeOutOfMemoryException(self.source, 'DataLoader', self.visitor)
        elif faultType == 'delay':
            return injections.addDelay(self.source, 'cosine_similarity', self.visitor)
        else:
            print('Fault Type is not supported.')