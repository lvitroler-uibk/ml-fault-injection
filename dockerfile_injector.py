import os

from shared.utils import write_new_source_code
from shared.utils import getIndexOfStringInList
from shared.utils import change_line_source

class DockerfileInjector:
    def __init__(self, fileName):
        source = open(fileName, 'r').readlines()
        raw_source = ''.join(source)
        REQUIREMENTS = 'requirements.'

        if REQUIREMENTS in raw_source:
            _, tail = os.path.split(fileName)
            fileName = fileName.replace(tail, REQUIREMENTS + 'in')

        self.fileName = fileName
        self.source = open(fileName, 'r').readlines()
        raw_source = ''.join(self.source)
        self.libraries = {}

        if 'torch' in raw_source:
            self.libraries['torch'] = '1.4.0'
            self.libraries['torchvision'] = '0.5.0'
            self.libraries['torchaudio'] = '0.4.0'
        elif 'keras' in raw_source:
            self.libraries['tensorflow'] = '1.15.2'
            self.libraries['keras'] = '2.3.1'
        elif 'tensorflow' in raw_source:
            self.libraries['tensorflow'] = '1.15.2'
    
    def deleteLibraries(self):
        newSource = None

        for library in self.libraries:
            if newSource is None:
                newSource = self.source

            index = getIndexOfStringInList(newSource, library)
            if index >= 0:
                newSource[index] = ''

        return newSource

    def changeLibraries(self):
        newSource = None

        for library in self.libraries:
            if newSource is None:
                newSource = self.source

            index = getIndexOfStringInList(newSource, library)
            if index >= 0:
                versionNumber = newSource[index][newSource[index].rfind('=') + 1:newSource[index].rfind('\n')]
                newSource = change_line_source(
                    newSource,
                    index,
                    newSource[index][newSource[index].rfind('=') + 1:],
                    newSource[index][newSource[index].rfind('=') + 1:].replace(versionNumber, self.libraries[library])
                )

        return newSource
        
    def inject(self, fault_type):
        source = None

        if 'delete' in fault_type:
            source = self.deleteLibraries()
        elif 'change' in fault_type:
            source = self.changeLibraries()

        if source:
            print('File has been changed')
            write_new_source_code(self.fileName, source, '')