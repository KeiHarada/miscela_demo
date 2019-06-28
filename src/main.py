import argparse
from src.func import miscela
from src.func import mocServer

if __name__ == "__main__":

    '''
    デモ用に改良済
    バグがあれば原田まで
    '''

    '''
    :parameters
        0. path_root_src
        1. dataset
        2. maxAtt
        3. minSup
        4. evoRate (from 0 to 1)
        5. distance ([km])
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_root_src", type=str, default="src/main.py")
    parser.add_argument("--dataset", help="which dataset would like to use", type=str, default="santander")
    parser.add_argument("--maxAtt", help="the maximum number of attributes you would like to find", type=int, default=2)
    parser.add_argument("--minSup", help="the minimum number of timestamps for co-evolution", type=int, default=1000)
    parser.add_argument("--evoRate", help="evolving rate", type=float, default=0.5)
    parser.add_argument("--distance", help="distance threshold", type=float, default=0.1)
    args = parser.parse_args()

    print(args)
    miscela(args)
    #mocServer(args)