import parsl
from parsl import *
import random
import argparse

workers = ThreadPoolExecutor(max_workers = 10)
dfk = DataFlowKernel(workers)

@App('python', dfk)
def rand():
    return random.randint(1, 10)

@App('python', dfk)
def square(x):
    return x**2

@App('python', dfk)
def cubed(x):
    return x**3

def test_discriminator(x = 4):
    squares = []
    cubes = []
    total = []
    for i in range(x):
        squares.append(square(rand().result()))
        cubes.append(cubed(rand().result()))
        while squares[i].done() == False and cubes[i].done() == False:
            pass
        total.append(squares[i].result() + cubes[i].result())
    for i in range(x):
        print(total[i])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--num1", default = "4", action = "store", dest = "x", type = int)
    args = parser.parse_args()
    test_discriminator(args.x)
