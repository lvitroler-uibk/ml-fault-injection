import ast

class Injector:
    
    def __init__(self, fileName):
        self.source = open(fileName, 'r').readlines()
        self.raw_source = ''.join(self.source)
        root_node = ast.parse(self.raw_source, fileName)

    def inject(self, fault_type):
        print('Hi')