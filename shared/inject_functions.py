from shared.utils import change_line_source
from shared.ast_parser import Visitor

def causeOutOfMemoryException(source, searchString, visitor: Visitor):
    funcLines = [idx for idx, s in enumerate(source) if searchString in s]
    if len(funcLines) == 0:
        return None
    
    newSource = None
    batchSizeName = 'batch_size'
    index = funcLines[0]
    func = visitor.lineno_function_call[index + 1][0]

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
        varLines = [idx for idx, s in enumerate(source) if varValue in s]
        varLines.reverse()
        
        for line in varLines:
            var = visitor.lineno_varname[line + 1][0]
            if var == varValue:
                varvalue = visitor.lineno_varvalue[line + 1][0]
                newSource = change_line_source(
                    source,
                    line,
                    str(varvalue.value),
                    str(int(varvalue.value) * int(varvalue.value))
                )
                break

    return newSource