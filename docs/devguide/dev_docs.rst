Developer Documentation
***********************


.. automodule:: parsl
   :no-undoc-members:

.. autofunction:: set_stream_logger

.. autofunction:: set_file_logger

Apps
====

Apps are parallelized functions that execute independent of the control flow of the main python
interpretor. We have two main types of Apps : PythonApps and BashApps. These are subclassed from
AppBase.

AppBase
-------

This is the base class that defines the two external facing functions that an App must define.
The  __init__ () which is called when the interpretor sees the definition of the decorated
function, and the __call__ () which is invoked when a decorated function is called by the user.

.. autoclass:: parsl.app.app.AppBase
   :members: __init__, __call__

PythonApp
---------

Concrete subclass of AppBase that implements the Python App functionality.

.. autoclass:: parsl.app.app.PythonApp
   :members: __init__, __call__

BashApp
-------

Concrete subclass of AppBase that implements the Bash App functionality.

.. autoclass:: parsl.app.app.BashApp
   :members: __init__, __call__


Futures
=======

Futures are returned as proxies to a parallel execution initiated by a call to an ``App``.
We have two kinds of apps: AppFutures and DataFutures.


AppFutures
----------

.. autoclass:: parsl.dataflow.futures.AppFuture
   :members:


DataFutures
-----------

.. autoclass:: parsl.app.futures.DataFuture
   :members:


Exceptions
==========

.. autoclass:: parsl.app.errors.ParslError

.. autoclass:: parsl.app.errors.NotFutureError

.. autoclass:: parsl.app.errors.InvalidAppTypeError

.. autoclass:: parsl.app.errors.AppException

.. autoclass:: parsl.dataflow.error.DataFlowExceptions

.. autoclass:: parsl.dataflow.error.DuplicateTaskError

.. autoclass:: parsl.dataflow.error.MissingFutError


Executors
=========

Executors are abstractions that represent compute resources to which you could submit arbitrary App tasks. These resources
themselves can (sometimes) scale to fit demand better.

We currently have thread pools, remote workers from `ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`_, and an
incomplete Swift/T executor for HPC systems.




Swift/Turbine Executor
----------------------

.. autoclass:: parsl.executors.swift_t.TurbineExecutor
   :members: _queue_management_worker, weakred_cb, _start_queue_management_thread, shutdown, __init__, submit, scale_out, scale_in

.. autofunction:: parsl.executors.swift_t.runner

                  
