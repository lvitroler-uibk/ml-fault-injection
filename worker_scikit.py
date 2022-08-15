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

    def inject(self, faultType):
        if faultType == 'memory':
            return self.causeOutOfMemoryException()
        elif faultType == 'delay':
            return injections.addDelay(self.source, 'cosine_similarity', self.visitor)
        elif faultType == 'API':
            return self.causeApiMismatch()
        else:
            print('Fault Type is not supported.')