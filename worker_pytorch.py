from shared.ast_parser import Visitor

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
            newValue = int(varValue) * int(varValue)
            lineno = batchSizeParam.start_lineno - 1
            newSource = self.source
            newSource[lineno] = self.source[lineno].replace(batchSizeName + '=' + varValue, batchSizeName + '=' + str(newValue))
        else:
            varLines = [idx for idx, s in enumerate(self.source) if varValue in s]
            varLines.reverse()
            
            for line in varLines:
                var = self.visitor.lineno_varname[line + 1][0]
                if var == varValue:
                    varvalue = self.visitor.lineno_varvalue[line + 1][0]
                    newValue = int(varvalue.value) * int(varvalue.value)
                    newSource = self.source
                    newSource[line] = self.source[line].replace(str(varvalue.value), str(newValue))
                    break
        return newSource

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()

        print(faultType)