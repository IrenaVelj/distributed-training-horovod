# distributed-training-horovod
In this repo, distributed training of neural networks is implemented and tested. The focus is on distributed training using Horovod.

# How to run python script using Horovod
In order to run training using Horovod, we need to use MPI protocol, and this we can do by using mpirun command.
Example:

    mpirun --np {NUMBER_OF_PROCESSES} python {NAME_OF_THE_PYTHON_SCRIPT}
