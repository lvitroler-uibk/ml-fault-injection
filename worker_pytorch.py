from shared.utils import change_line_source
import random
import shared.inject_functions as injections
from shared.ast_parser import Visitor

class WorkerPyTorch:
    
    def __init__(self, source, visitor: Visitor):
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


    def changeModelLoad(self):
        funcs = injections.getFuncs(self.visitor, 'torch.load')
        if len(funcs) == 0:
            return None

        func = funcs[len(funcs) - 1]
        newSource = self.source
        loadString = newSource[func.lineno - 1]
        indentationLength = len(loadString) - len(loadString.lstrip())

        modelName = 'model_faulty.pt'
        modelFile = 'https://github.com/lvitroler-uibk/ml-fault-injection/raw/main/example_models/pytorch/model_fine_tuned.pt'
        loadString = "torch.hub.download_url_to_file('" + modelFile + "', '" + modelName + "')"
        newSource.insert(func.lineno - 1, loadString.rjust(len(loadString) + indentationLength) + '\n')

        fun_params = self.visitor.func_key_raw_params[func]
        _, rawParam = fun_params[0]
        newSource = change_line_source(
            newSource,
            func.lineno,
            rawParam.name,
            "'" + modelName + "'"
        )

        return newSource

    def halfTestData(self):
        funcs = injections.getFuncs(self.visitor, 'DataLoader')
        if len(funcs) == 0:
            return None

        newSource = self.source
        counter = 0
        for func in funcs:
            if counter > 1:
                break

            fun_params = self.visitor.func_key_raw_params[func]
            _, rawParam = fun_params[0]

            newSource = change_line_source(
                newSource,
                rawParam.start_lineno - 1,
                rawParam.name,
                rawParam.name + '/2'
            )        
            counter += 1
        
        return newSource


    def inject(self, faultType):
        loss_functions = [
            'nll_loss',
            'poisson_nll_loss',
            'gaussian_nll_loss',
            'kl_div',
            'cross_entropy',
            'binary_cross_entropy',
            'binary_cross_entropy_with_logits',
            'smooth_l1_loss',
        ];

        # Für model stage
        #falsches bild bzw. falsches model laden in github gespeichert --> change hyperparams

        #anderes neurales Netzwerk verwenden (z.B. statt resnet18 was anderes)
        #class_names leer machen

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
        elif faultType == 'PRI':
            return injections.causeParameterRestrictionIncompatible(self.source, 'unsqueeze', self.visitor)
        elif faultType == 'GPU':
            return self.causeGpuUsageMismatch()
        elif faultType == 'optim':
            return self.changeOptimiser()
        elif faultType == 'hyperparams':
            return injections.worsenHyperparameters(self.source, 'DataLoader', self.visitor)
        elif faultType == 'model':
            return self.changeModelLoad()
        elif faultType == 'breakmodel':
            return injections.breakModelLoad(self.source, 'torch.load', self.visitor)
        elif faultType == 'delay':
            return injections.addDelay(self.source, 'model', self.visitor)
        elif faultType == 'normalisation':
            return injections.removeNormalisation(self.source, 'Normalize', self.visitor)
        elif faultType == 'networks':
            networkSwitches = {
                'Module': 'RNN',
                'RNN': 'Module'
            }

            return injections.changeNetworks(self.source, networkSwitches, self.visitor)
        elif faultType == 'datatype':
            return injections.changeDataType(self.source, self.visitor)
        elif faultType == 'edittestdata':
            return self.halfTestData()
        else:
            print('Fault Type is not supported.')