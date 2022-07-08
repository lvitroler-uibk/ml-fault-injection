import sys

from regex import D
from python_injector import PythonInjector
from dockerfile_injector import DockerfileInjector

if __name__ == '__main__':
    break_repair = sys.argv[1] # break or repair
    fault_type = sys.argv[2]
    fileName = sys.argv[3]

    try:
        if break_repair == 'python':
            injector = PythonInjector(fileName)
            injector.inject(fault_type)
        elif break_repair == 'dockerfile':
            injector = DockerfileInjector(fileName)
            injector.inject(fault_type)
        else:
            print('Wrong parameter')
    except Exception as e:
        print('ERROR: ' + str(e))
        raise e