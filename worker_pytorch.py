from shared.ast_parser import Visitor

class WorkerPyTorch:
    
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        lines = [idx for idx, s in enumerate(self.source) if 'DataLoader(' in s]
        if len(lines) == 0:
            return None
        
        newSource = None
        batchSizeName = 'batch_size'
        index = lines[0]
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
        
        return newSource

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()

        print(faultType)