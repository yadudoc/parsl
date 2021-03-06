''' Testing behavior of futures

We have the following cases for AppFutures:

1. App launched immmediately on call
2. App launch was delayed (due to unresolved dep(s))

Same applies to datafutures, and we need to know the behavior wrt.

1. result() called on 1, vs 2
2. done() called on 1, vs 2

'''
import parsl
from parsl import *

#from nose.tools import nottest
import os
import time
import shutil
import argparse

workers = ThreadPoolExecutor(max_workers=8)
#parsl.set_stream_logger()
#workers = ProcessPoolExecutor(max_workers=4)
dfk = DataFlowKernel(workers)


@App('python', dfk)
def delay_incr(x, delay=0, outputs=[]):
    import time
    import os
    if outputs :
        with open(outputs[0], 'w') as outs:
            outs.write(str(x+1))
    time.sleep(delay)
    return x+1

def get_contents(filename):
    with open(filename, 'r') as f:
        return f.read()

def test_fut_case_1():
    ''' Testing the behavior of AppFutures where there are no dependencies
    '''

    app_fu = delay_incr(1, delay=0.5)

    status = app_fu.done()
    result = app_fu.result()

    print("Status : ", status)
    print("Result : ", result)

    assert result == 2, 'Output does not match expected 2, goot: "{0}"'.format(result)
    return True

def test_fut_case_2():
    ''' Testing the behavior of DataFutures where there are no dependencies
    '''
    output_f = 'test_fut_case_2.txt'
    app_fu, [data_fu] = delay_incr(1, delay=0.5, outputs=[output_f])

    status = data_fu.done()
    result = data_fu.result()
    print ("App_fu  : ", app_fu)
    print ("Data_fu : ", data_fu)


    assert os.path.basename(result) == output_f , \
        "DataFuture did not return the filename, got : {0}".format(result)
    print("Status : ", status)
    print("Result : ", result)

    contents = get_contents(result)
    assert contents == '2', 'Output does not match expected "2", got: "{0}"'.format(result)
    return True

def test_fut_case_3():
    ''' Testing the behavior of AppFutures where there are dependencies

    The first call has a delay of 0.5s, and the second call depends on the first
    '''

    app_1 = delay_incr(1, delay=0.5)
    app_2 = delay_incr(app_1)

    status = app_2.done()
    result = app_2.result()

    print("Status : ", status)
    print("Result : ", result)

    assert result == 3, 'Output does not match expected 2, goot: "{0}"'.format(result)
    return True

def test_fut_case_4():
    ''' Testing the behavior of DataFutures where there are dependencies

    The first call has a delay of 0.5s, and the second call depends on the first
    '''
    ''' Testing the behavior of DataFutures where there are no dependencies
    '''
    output_f = 'test_fut_case_4.txt'
    app_1, [data_1] = delay_incr(1, delay=0.5, outputs=[output_f])
    app_2, [data_2] = delay_incr(app_1, delay=0.5, outputs=[output_f])

    status = data_2.done()
    result = data_2.result()
    print ("App_fu  : ", app_2)
    print ("Data_fu : ", data_2)


    assert os.path.basename(result) == output_f , \
        "DataFuture did not return the filename, got : {0}".format(result)
    print("Status : ", status)
    print("Result : ", result)

    contents = get_contents(result)
    assert contents == '3', 'Output does not match expected "3", got: "{0}"'.format(result)
    return True


if __name__ == '__main__' :

    parser   = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", default="10", help="Count of apps to launch")
    parser.add_argument("-d", "--debug", action='store_true', help="Count of apps to launch")
    args   = parser.parse_args()

    if args.debug:
        parsl.set_stream_logger()

    #x = test_parallel_for(int(args.count))
    y = test_fut_case_3() 
    #raise_error(0)
