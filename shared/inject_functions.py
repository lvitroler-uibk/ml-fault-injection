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

def causeOutOfMemoryException(source, searchString, visitor: Visitor):
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
            str(int(varValue) * int(varValue))
        )
    else:
        varLines = getVars(visitor, varValue)
        varLines.reverse()
        print(varValue)
        
        for line in varLines:
            var = visitor.lineno_varname[line][0]
            if var == varValue:
                varvalue = visitor.lineno_varvalue[line][0]
                newSource = change_line_source(
                    source,
                    line - 1,
                    str(varvalue.value),
                    str(int(varvalue.value) * int(varvalue.value))
                )
                break

    return newSource

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

def injectFiiModelInputShape(source, searchString, visitor: Visitor):
    funcs = getFuncs(visitor, searchString)
    if len(funcs) == 0:
        return None
    
    newSource = None
    for func in funcs:
        funParams = visitor.func_key_raw_params[func]
        for name, rawParam in funParams:
            if name is None or 'input_dim' not in name:
                continue

            funcStartPos = rawParam.start_index
            oldString = source[func.lineno - 1][funcStartPos:]

            newSource = change_line_source(
                source,
                func.lineno - 1,
                oldString,
                oldString.replace(str(rawParam.name), str(int(rawParam.name) + int(rawParam.name)))
            )
            break

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

