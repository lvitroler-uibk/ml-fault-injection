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
        previousFuncLineNo = -1
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

    def causeGpuUsageMismatch(self):
        funcTo = '.to'
        funcs = injections.getFuncs(self.visitor, funcTo)
        if len(funcs) == 0:
            return None

        newSource = self.source
        previousFuncLineNo = -1

        for func in funcs:
            lineNo = func.lineno - 1
            startIndex = func.start_index
            if previousFuncLineNo == lineNo:
                lineString = newSource[lineNo][startIndex:]

                newSource = change_line_source(
                    newSource,
                    lineNo,
                    lineString[:lineString.find(')') + 1],
                    lineString[:lineString.find(funcTo)]
                )
                break

            previousFuncLineNo = lineNo

        return newSource

    def changeOptimiser(self):
        optimisers = injections.getListOfOptimisers()
        for optim in optimisers:
            funcs = injections.getFuncs(self.visitor, optim)
            if len(funcs) > 0:
                break

        if len(funcs) == 0:
            return None
        
        optimisers.remove(optim)
        random.shuffle(optimisers)

        return change_line_source(
            self.source,
            funcs[0].lineno -1,
            optim,
            optimisers[0]
        )

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
        elif faultType == 'GPU':
            return self.causeGpuUsageMismatch()
        elif faultType == 'optim':
            return self.changeOptimiser()
        else:
            print('Fault Type is not supported.')