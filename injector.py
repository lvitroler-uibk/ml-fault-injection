import ast
from shared.utils import write_new_source_code
from shared.ast_parser import Visitor
from worker_keras import WorkerKeras
from worker_pytorch import WorkerPyTorch
from worker_tensorflow import WorkerTensorflow

class Injector:
    def __init__(self, fileName):
        self.fileName = fileName
        self.source = open(fileName, 'r').readlines()
        raw_source = ''.join(self.source)
        self.root_node = ast.parse(raw_source, fileName)

        if 'torch' in raw_source:
            self.library = 'torch'
        elif 'keras' in raw_source:
            self.library = 'keras'
        elif 'tensorflow' in raw_source:
            self.library = 'tensorflow'
        
        self.visitor = Visitor(self.source)
        self.visitor.visit(self.root_node)

    def inject(self, fault_type):
        if 'torch' in self.library:
            workerTorch = WorkerPyTorch(self.source, self.visitor)
            source = workerTorch.inject(fault_type)
        elif 'keras' in self.library:
            workerKeras = WorkerKeras(self.source, self.visitor)
            source = workerKeras.inject(fault_type)
        elif 'tensorflow' in self.library:
            workerTensorflow = WorkerTensorflow(self.source, self.visitor)
            source = workerTensorflow.inject(fault_type)

        if source:
            write_new_source_code(self.fileName, source)
