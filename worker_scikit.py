from imp import source_from_cache
from shared.utils import change_line_source
import random
import shared.inject_functions as injections
from shared.ast_parser import Visitor

class WorkerScitkit:
    def __init__(self, source, visitor: Visitor):
        self.source = source
        self.visitor = visitor

    def causeOutOfMemoryException(self):
        pandaFunctions = [
            '.json_normalize',
            '.read_json',
            '.read_csv'
        ]

        newSource = None
        for function in pandaFunctions:
            funcs = injections.getFuncs(self.visitor, function)
            if len(funcs) == 0:
                continue

            func = funcs[0]
            newSource = change_line_source(
                self.source,
                func.lineno - 1,
                '\n',
                '* 200000000000000000\n'
            )
            break

        return newSource

    def causeApiMismatch(self):
        textFunctions = [
            'CountVectorizer',
            'TfidfVectorizer',
            'HashingVectorizer'
        ]

        newSource = None
        for function in textFunctions:
            funcs = injections.getFuncs(self.visitor, function)
            if len(funcs) == 0:
                continue

            func = funcs[0]
            newSource = change_line_source(
                self.source,
                func.lineno -1,
                '(',
                '(charset=None,'
            )

        return newSource

    def causeDataFormatFault(self):
        funcs = injections.getFuncs(self.visitor, 'fit_transform')
        if len(funcs) == 0:
            return None
        
        func = funcs[0]
        fun_params = self.visitor.func_key_raw_params[func]
        _, rawParam = fun_params[0]

        return change_line_source(
                self.source,
                rawParam.start_lineno - 1,
                self.source[rawParam.start_lineno - 1],
                rawParam.name + ' = 1\n' + self.source[rawParam.start_lineno - 1]
            )

    def causeDataInitFault(self):
        funcs = injections.getFuncs(self.visitor, 'fit_transform')
        if len(funcs) == 0:
            return None
        
        func = funcs[0]
        fun_params = self.visitor.func_key_raw_params[func]
        _, rawParam = fun_params[0]

        return change_line_source(
                self.source,
                rawParam.start_lineno - 1,
                self.source[rawParam.start_lineno - 1],
                rawParam.name + ' = "key"\n' + self.source[rawParam.start_lineno - 1]
            )

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        elif faultType == 'delay':
            return injections.addDelay(self.source, 'cosine_similarity', self.visitor)
        elif faultType == 'API':
            return self.causeApiMismatch()
        elif faultType == 'dataformat':
            return self.causeDataFormatFault()
        elif faultType == 'datainit':
            return self.causeDataInitFault()
        else:
            print('Fault Type is not supported.')