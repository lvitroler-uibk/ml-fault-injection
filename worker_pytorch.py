from shared.ast_parser import Visitor
from shared.utils import change_line_source

class WorkerPyTorch:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        funcLines = [idx for idx, s in enumerate(self.source) if 'DataLoader(' in s]
        if len(funcLines) == 0:
            return None
        
        newSource = None
        batchSizeName = 'batch_size'
        index = funcLines[0]
        func = self.visitor.lineno_function_call[index + 1][0]

        fun_params = self.visitor.func_key_raw_params[func]
        
        batchSizeParam = None
        for varName, rawParam in fun_params:
            if varName is not None and batchSizeName in varName:
                batchSizeParam = rawParam
                break

        varValue = batchSizeParam.name
        if varValue.isdigit():
            newSource = change_line_source(
                self.source,
                batchSizeParam.start_lineno - 1,
                batchSizeName + '=' + varValue,
                batchSizeName + '=' + str(int(varValue) * int(varValue))
            )
        else:
            varLines = [idx for idx, s in enumerate(self.source) if varValue in s]
            varLines.reverse()
            
            for line in varLines:
                var = self.visitor.lineno_varname[line + 1][0]
                if var == varValue:
                    varvalue = self.visitor.lineno_varvalue[line + 1][0]
                    newSource = change_line_source(
                        self.source,
                        line,
                        str(varvalue.value),
                        str(int(varvalue.value) * int(varvalue.value))
                    )
                    break

        return newSource

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        else:
            print('Fault Type is not supported.')