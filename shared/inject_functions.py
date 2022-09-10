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

def getLoads(visitor: Visitor, searchString):
    foundVars = []

    for loads in visitor.load:
        for key, value in loads.items():
            if searchString in key:
                foundVars.append(value)

    return foundVars

def changeParamValue(source, searchString, visitor: Visitor, batchSizeMultiplier, paramName):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = None
    func = funcs[0]

    fun_params = visitor.func_key_raw_params[func]
    
    batchSizeParam = None
    for varName, rawParam in fun_params:
        if varName is not None and paramName in varName:
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
    return changeParamValue(source, searchString, visitor, 100000, 'batch_size')

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

        oldValue = rawParam.name
        if oldValue.isdigit():
            newString = str(int(oldValue) + int(oldValue))
        else:
            newString = oldValue + ' + ' + oldValue

        newSource = change_line_source(
            newSource,
            func.lineno - 1,
            oldString,
            oldString.replace(str(rawParam.name), newString)
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
    newSource = changeParamValue(source, searchString, visitor, 0.5, 'batch_size')

    return changeParamValue(newSource, searchString, visitor, 0.5, 'epochs')

def changeModelLoad(source, visitor: Visitor):
    funcs = getFuncs(visitor, 'load_weights')
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]
    newSource = source

    modelString = newSource[func.lineno - 1]
    indentationLength = len(modelString) - len(modelString.lstrip())

    weightsPath = 'weights_path'
    modelFile = 'https://github.com/lvitroler-uibk/ml-fault-injection/raw/main/example_models/keras/weights.h5'
    newSource.insert(0, 'from keras.utils.data_utils import get_file\n')

    loadString = weightsPath + " = get_file('model_faulty', '" + modelFile + "')"
    newSource.insert(func.lineno, loadString.rjust(len(loadString) + indentationLength) + '\n')

    fun_params = visitor.func_key_raw_params[func]
    
    _, rawParam = fun_params[0]
    newSource = change_line_source(
        newSource,
        func.lineno + 1,
        rawParam.name,
        weightsPath
    )

    return newSource

def breakModelLoad(source, loadString, visitor: Visitor):
    funcs = getFuncs(visitor, loadString)
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]
    fun_params = visitor.func_key_raw_params[func]
    _, rawParam = fun_params[0]
    rawParamName = rawParam.name
    newString = rawParamName

    if rawParamName.find('/') != -1:
        newString = newString.replace('/', '../', 1)
    else:
        newString = newString.replace("'", "'../", 1)

    newSource = change_line_source(
        source,
        func.lineno - 1,
        rawParamName,
        newString
    )

    return newSource

def addDelay(source, predictString, visitor: Visitor):
    funcs = getFuncs(visitor, predictString)
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]
    newSource = source
    modelString = newSource[func.lineno - 1]
    indentationLength = len(modelString) - len(modelString.lstrip())
    newSource.insert(0, 'import time\n')
    sleepString = 'time.sleep(2)'
    newSource.insert(func.lineno, sleepString.rjust(len(sleepString) + indentationLength) + '\n')

    return newSource

def removeNormalisation(source, normalisationString, visitor: Visitor):
    funcs = getFuncs(visitor, normalisationString)
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]
    newSource = source
    newSource[func.lineno - 1] = ''

    return newSource

def changeNetworks(source, networkSwitches, visitor: Visitor):
    for key, value in networkSwitches.items():
        loads = getLoads(visitor, key)
        if len(loads) == 0:
            continue

        load = loads[len(loads) - 1]
        newSource = change_line_source(
            source,
            load.lineno - 1,
            key,
            value
        )

    return newSource

def changeDataType(source, visitor: Visitor):
    dataTypes = [
        'float',
        'int'
    ]

    dataTypeFound = False
    newSource = None
    for index, dataType in enumerate(dataTypes):
        if dataTypeFound:
            break

        for _, vars in visitor.func_key_raw_params.items():
            for _, rawParam in vars:
                if dataType not in rawParam.name:
                    continue
                if newSource == None:
                    newSource = source

                copy = dataTypes.copy()
                copy.pop(index)
                random.shuffle(copy)

                newSource = change_line_source(
                    newSource,
                    rawParam.start_lineno - 1,
                    dataType,
                    copy[0]
                )

    return newSource

def halfTestData(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    func = funcs[len(funcs) - 1]
    fun_params = visitor.func_key_raw_params[func]

    newSource = source
    counter = 0
    for _, rawParam in fun_params:
        if counter > 1:
            break
        
        newSource = change_line_source(
            newSource,
            rawParam.start_lineno - 1,
            rawParam.name,
            rawParam.name + '/2'
        )        
        counter += 1
    
    return newSource

def causeParameterRestrictionIncompatible(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    func = funcs[len(funcs) - 1]
    newSource = source
    newSource[func.lineno -1] = ''

    return newSource

def switchFunctions(source, switchFunctions, visitor: Visitor):
    for key, value in switchFunctions.items():
        funcs = getFuncs(visitor, key)
        if len(funcs) == 0:
            continue

        func = funcs[len(funcs) - 1]
        return change_line_source(
            source,
            func.lineno - 1,
            key,
            value
        )

    return None

def addFaultyCondition(source, predictString, visitor: Visitor):
    funcs = getFuncs(visitor, predictString)
    if len(funcs) == 0:
        return None

    func = funcs[len(funcs) - 1]

    newSource = source
    
    for i in range(func.lineno, func.lineno+10):
        if i >= len(source):
            break

        line = newSource[i]
        if 'class' in line:
            subString = line[line.find('class'):]

            newSource = change_line_source(
                source,
                i,
                subString[subString.find('['):subString.find(']') + 1],
                '[0]'
            )
            indentationLength = len(newSource[i]) - len(newSource[i].lstrip())
            conditionString = 'if True:'
            newSource.insert(i, conditionString.rjust(len(conditionString) + indentationLength) + '\n  ')

            return newSource

    return None