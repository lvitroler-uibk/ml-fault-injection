import sys

from injector import Injector



if __name__ == '__main__':
    break_repair = sys.argv[1] # break or repair
    fault_type = sys.argv[2]
    fileName = sys.argv[3]

#    break_repair = 'break'
#    fault_type = 'memory'
#    fileName = '../microservices-demo/src/classifyingservice/train.py'
    
    try:
        if break_repair == 'break':
            injector = Injector(fileName)
            injector.inject(fault_type)
    except Exception as e:
        print('ERROR: ' + str(e))