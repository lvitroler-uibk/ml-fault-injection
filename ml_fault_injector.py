import sys

from injector.injector import Injector



if __name__ == '__main__':
    break_repair = sys.argv[1] # break or repair
    fault_type = sys.argv[2]
    file = sys.argv[3]
    
    source = open(file, 'r').readlines()

    injector = Injector(source)
    injector.inject(fault_type)