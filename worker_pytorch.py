from shared.utils import change_line_source
import random
import shared.inject_functions as injections

class WorkerPyTorch:
    
    def __init__(self, source, visitor):
        self.source = source
        self.visitor = visitor

    def causeFeatureInputIncompatible(self):
        possibleFaults = [
            'modelInputShape'
        ]
        random.shuffle(possibleFaults)
        newSource = None

        for fault in possibleFaults:
            if fault == 'modelInputShape':
                newSource = injections.injectFiiModelInputShape(self.source, 'Linear', self.visitor, False)

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
                newSource = injections.causeLoiModelOutputShape(self.source, 'Linear', self.visitor)

            if newSource is not None:
                break

        return newSource
    
    # https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
    def causeApiMismatch(self):
        funcs = injections.getFuncs(self.visitor, '.to')
        if len(funcs) == 0:
            return None

        newSource = self.source
        previousFuncLineNo = 0
        offset = 0

        for func in funcs:
            lineNo = func.lineno - 1
            startIndex = func.start_index
            if previousFuncLineNo == lineNo:
                startIndex += offset
            
            lineString = newSource[lineNo][startIndex:]
            oldString = lineString[:lineString.find(')') + 1]
            newString = 'Variable(' + lineString[:lineString.find('.')] + ')'
            offset = len(newString) - len(oldString)

            newSource = change_line_source(
                newSource,
                lineNo,
                oldString,
                newString
            )
            previousFuncLineNo = lineNo

        return newSource

    def inject(self, faultType):
        if faultType == 'memory':
            return injections.causeOutOfMemoryException(self.source, 'DataLoader', self.visitor)
        elif faultType == 'FII':
            return self.causeFeatureInputIncompatible()
        elif faultType == 'ALI':
            return injections.causeAdjacentLayerIncompatible(self.source, 'flatten', self.visitor)
        elif faultType == 'LOI':
            return self.causeLabelOutputIncompatible()
        elif faultType == 'API':
            return self.causeApiMismatch()
        else:
            print('Fault Type is not supported.')