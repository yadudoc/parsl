from parsl.config import Config
from parsl.executors import WorkQueueExecutor

from parsl.data_provider.http import HTTPInTaskStaging
from parsl.data_provider.ftp import FTPInTaskStaging
from parsl.data_provider.file_noop import NoOpFileStaging
from parsl.monitoring import MonitoringHub


def fresh_config():
    return Config(executors=[WorkQueueExecutor(port=9000,
                                               coprocess=True,
                                               storage_access=[FTPInTaskStaging(), HTTPInTaskStaging(), NoOpFileStaging()])],
                  monitoring=MonitoringHub(hub_address="localhost",
                                           hub_port=55055,
                                           monitoring_debug=True,
                                           resource_monitoring_interval=1,
                                           ))
