import ast
from shared.utils import write_new_source_code
from shared.ast_parser import Visitor

class Injector:
    
    def __init__(self, fileName):
        self.fileName = fileName
        self.source = open(fileName, 'r').readlines()
        self.raw_source = ''.join(self.source)
        self.root_node = ast.parse(self.raw_source, fileName)

        if 'torch' in self.raw_source:
            self.library = 'torch'
        elif 'keras' in self.raw_source:
            self.library = 'keras'
        elif 'tensorflow' in self.raw_source:
            self.library = 'tensorflow'
        
        self.visitor = Visitor(self.source)
        self.visitor.visit(self.root_node)


    def inject(self, fault_type):
        print(self.library)
        write_new_source_code(self.fileName, self.source)
