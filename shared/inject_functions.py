import random

from shared.utils import change_line_source
from shared.ast_parser import Visitor

def getFuncs(visitor: Visitor, searchString):
    foundFuncs = []

    for _, funcs in visitor.lineno_function_call.items():
        for func in funcs:
            if searchString in func.name:
                foundFuncs.append(func)

    return foundFuncs

def getVars(visitor: Visitor, searchString):
    foundVars = []

    for lineno, vars in visitor.lineno_varname.items():
        for var in vars:
            if searchString in var:
                foundVars.append(lineno)

    return foundVars

def changeBatchSize(source, searchString, visitor: Visitor, batchSizeMultiplier):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = None
    batchSizeName = 'batch_size'
    func = funcs[0]

    fun_params = visitor.func_key_raw_params[func]
    
    batchSizeParam = None
    for varName, rawParam in fun_params:
        if varName is not None and batchSizeName in varName:
            batchSizeParam = rawParam
            break

    varValue = batchSizeParam.name
    if varValue.isdigit():
        newSource = change_line_source(
            source,
            batchSizeParam.start_lineno - 1,
            varValue,
            str(int(int(varValue) * batchSizeMultiplier))
        )
    else:
        varLines = getVars(visitor, varValue)
        varLines.reverse()

        for line in varLines:
            var = visitor.lineno_varname[line][0]
            if var == varValue:
                varvalue = visitor.lineno_varvalue[line][0]
                newSource = change_line_source(
                    source,
                    line - 1,
                    str(varvalue.value),
                    str(int(int(varvalue.value) * batchSizeMultiplier))
                )
                break

    return newSource


def causeOutOfMemoryException(source, searchString, visitor: Visitor):
    return changeBatchSize(source, searchString, visitor, 100000)

def injectFiiExpandDims(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = source
    funcs.reverse()
    for func in funcs:
        if searchString not in func.name:
            continue

        funParams = visitor.func_key_raw_params[func]

        funcStartPos = func.start_index
        funcEndPos = newSource[func.lineno - 1].find(')', funcStartPos) + 1
        _, rawParam = funParams[0]
        funcFirstVar = rawParam.name

        newSource = change_line_source(
        newSource,
        func.lineno - 1,
        newSource[func.lineno - 1][funcStartPos:funcEndPos],
        funcFirstVar
)

    return newSource

def injectFiiModelInputShape(source, searchString, visitor: Visitor, checkForInputDim = True):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = None
    for func in funcs:
        funParams = visitor.func_key_raw_params[func]
        for name, rawParam in funParams:
            if checkForInputDim and (name is None or 'input_dim' not in name):
                continue

            funcStartPos = rawParam.start_index
            oldString = source[func.lineno - 1][funcStartPos:]
            oldValue = rawParam.name
            if not checkForInputDim:
                commaPos = oldString.find(',')
                oldString = oldString[commaPos + 1:]

            newSource = change_line_source(
                source,
                func.lineno - 1,
                oldString,
                oldString.replace(str(oldValue), str(int(oldValue) + int(oldValue)))
            )

    return newSource

def causeAdjacentLayerIncompatible(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = source
    for func in funcs:
        newSource = change_line_source(
            newSource,
            func.lineno - 1,
            newSource[func.lineno - 1],
            ''
        )

    return newSource

def causeLoiModelOutputShape(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = source
    for func in funcs:
        funParams = visitor.func_key_raw_params[func]
        _, rawParam = funParams[0]

        funcStartPos = rawParam.start_index
        oldString = newSource[func.lineno - 1][funcStartPos:]

        commaPos = oldString.find(',')
        elipsisPos = oldString.find(')')
        denseEndPos = commaPos
        if commaPos < 0 or commaPos > elipsisPos:
            denseEndPos = elipsisPos
        oldString = oldString[:denseEndPos + 1]

        newSource = change_line_source(
            newSource,
            func.lineno - 1,
            oldString,
            oldString.replace(str(rawParam.name), str(int(rawParam.name) + int(rawParam.name)))
        )

    return newSource

def exchangeParameterNames(source, methodName, exchanges, visitor: Visitor):
    funcs = getFuncs(visitor, methodName)
    if len(funcs) == 0:
        return None
    
    newSource = None
    func = funcs[len(funcs) - 1]

    fun_params = visitor.func_key_raw_params[func]
    
    for funParamName, exchange in exchanges.items():
        for name, param in fun_params:
            if name is None or funParamName not in name:
                continue

            newSource = change_line_source(
                source,
                param.start_lineno - 1,
                name,
                exchange
            )
            break

    return newSource

def getListOfOptimisers():
    return [
        'Adadelta',
        'Adagrad',
        "Adam",
        'Adamax',
        'NAdam',
        'RMSprop',
        'SGD',
        ]

def changeOptimiser(source, methodName, visitor: Visitor):
    funcs = getFuncs(visitor, methodName)
    if len(funcs) == 0:
        return None
    
    optimisers = getListOfOptimisers()
    newSource = None
    func = funcs[len(funcs) - 1]
    fun_params = visitor.func_key_raw_params[func]

    for name, param in fun_params:
        if name is None or 'optimizer' not in name:
            continue

        paramName = param.name.replace('"', '')
        loweredOptimisers = list(name.lower() for name in optimisers)
        
        optimisers.pop(loweredOptimisers.index(paramName.lower()))
        random.shuffle(optimisers)
        newSource = change_line_source(
                source,
                param.start_lineno - 1,
                paramName,
                optimisers[0]
            )
        break

    return newSource

def worsenHyperparameters(source, searchString, visitor: Visitor):
    return changeBatchSize(source, searchString, visitor, 0.5)

def changeModelLoad(source, loadName, visitor: Visitor):
    funcs = getFuncs(visitor, loadName)
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]
    newSource = source
    modelName = 'model_faulty.pt'
    modelFile = 'https://github.com/lvitroler-uibk/ml-fault-injection/raw/main/example_models/pytorch/model_fine_tuned.pt'
    newSource.insert(func.lineno - 1, "torch.hub.download_url_to_file('" + modelFile + "', '" + modelName + "')\n")
    fun_params = visitor.func_key_raw_params[func]
    
    _, rawParam = fun_params[0]
    newSource = change_line_source(
        newSource,
        func.lineno,
        rawParam.name,
        "'" + modelName + "'"
    )

    return newSource