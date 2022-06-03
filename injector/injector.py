class Injector:
    
    def __init__(self, source):
        self.source = source
        self.raw_source = ''.join(source)

    def inject(self, fault_type):
        print('Hi')