class WorkerPyTorch:
    
    def __init__(self, source, visitor):
        self.source = source
        self.visitor = visitor

    def inject(self, faultType):
        print(faultType)