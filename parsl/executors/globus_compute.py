from __future__ import annotations

import copy
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional, Union

from parsl.errors import OptionalModuleMissing
from parsl.executors.base import ParslExecutor
from parsl.utils import RepresentationMixin

try:
    from globus_compute_sdk import Client, Executor
    _globus_compute_enabled = True
except ImportError:
    _globus_compute_enabled = False

UUID_LIKE_T = Union[uuid.UUID, str]


class GlobusComputeExecutor(ParslExecutor, RepresentationMixin):
    """ GlobusComputeExecutor enables remote execution on Globus Compute endpoints

    GlobusComputeExecutor is a thin wrapper over globus_compute_sdk.Executor
    Refer to `globus-compute user documentation <https://globus-compute.readthedocs.io/en/latest/executor.html>`_
    and `reference documentation <https://globus-compute.readthedocs.io/en/latest/reference/executor.html>`_
    for more details.

    .. note::
       As a remote execution system, Globus Compute relies on serialization to ship
       tasks and results between the Parsl client side and the remote Globus Compute
       Endpoint side. Serialization is unreliable across python versions, and
       wrappers used by Parsl assume identical Parsl versions across on both sides.
       We recommend using matching Python, Parsl and Globus Compute version on both
       the client side and the endpoint side for stable behavior.

    """

    def __init__(
            self,
            endpoint_id: UUID_LIKE_T,
            task_group_id: Optional[UUID_LIKE_T] = None,
            resource_specification: Optional[Dict[str, Any]] = None,
            user_endpoint_config: Optional[Dict[str, Any]] = None,
            label: str = "GlobusComputeExecutor",
            batch_size: int = 128,
            amqp_port: Optional[int] = None,
            client: Optional[Client] = None,
            **kwargs,
    ):
        """
        Parameters
        ----------

        endpoint_id:
            id of the endpoint to which to submit tasks

        task_group_id:
            The Task Group to which to associate tasks.  If not set,
            one will be instantiated.

        resource_specification:
            Specify resource requirements for individual task execution.

        user_endpoint_config:
            User endpoint configuration values as described
            and allowed by endpoint administrators. Must be a JSON-serializable dict
            or None. Refer docs from `globus-compute
            <https://globus-compute.readthedocs.io/en/latest/endpoints/endpoints.html#templating-endpoint-configuration>`_
            for more info.

        label:
            a label to name the executor

        batch_size:
            the maximum number of tasks to coalesce before
            sending upstream [min: 1, default: 128]

        amqp_port:
            Port to use when connecting to results queue. Note that the
            Compute web services only support 5671, 5672, and 443.

        client:
            instance of globus_compute_sdk.Client to be used by the executor.
            If not provided, the executor will instantiate one with default arguments.

        kwargs:
            Other kwargs listed will be passed through to globus_compute_sdk.Executor
            as is. Refer to `globus-compute docs
            <https://globus-compute.readthedocs.io/en/latest/reference/executor.html#globus-compute-executor>`_
        """
        super().__init__()
        self.endpoint_id = endpoint_id
        self.task_group_id = task_group_id
        self.resource_specification = resource_specification
        self.user_endpoint_config = user_endpoint_config
        self.label = label
        self.batch_size = batch_size
        self.amqp_port = amqp_port
        self.client = client

        if not _globus_compute_enabled:
            raise OptionalModuleMissing(
                ['globus-compute-sdk'],
                "GlobusComputeExecutor requires globus-compute-sdk installed"
            )

        self._executor: Executor = Executor(
            endpoint_id=endpoint_id,
            task_group_id=task_group_id,
            resource_specification=resource_specification,
            user_endpoint_config=user_endpoint_config,
            label=label,
            batch_size=batch_size,
            amqp_port=amqp_port,
            client=self.client,
            **kwargs
        )

    def start(self) -> None:
        pass

    def submit(self, func: Callable, resource_specification: Dict[str, Any], *args: Any, **kwargs: Any) -> Future:
        """ Submit func to globus-compute


        Parameters
        ----------

        func: Callable
            Python function to execute remotely

        resource_specification: Dict[str, Any]
            Resource specification can be used specify MPI resources required by MPI applications on
            Endpoints configured to use globus compute's MPIEngine. GCE also accepts *user_endpoint_config*
            to configure endpoints when the endpoint is a `Multi-User Endpoint
            <https://globus-compute.readthedocs.io/en/latest/endpoints/endpoints.html#templating-endpoint-configuration>`_

        args:
            Args to pass to the function

        kwargs:
            kwargs to pass to the function

        Returns
        -------

        Future
        """
        res_spec = copy.deepcopy(resource_specification or self.resource_specification)
        # Pop user_endpoint_config since it is illegal in resource_spec for globus_compute
        if res_spec:
            user_endpoint_config = res_spec.pop('user_endpoint_config', self.user_endpoint_config)
        else:
            user_endpoint_config = self.user_endpoint_config

        self._executor.resource_specification = res_spec
        self._executor.user_endpoint_config = user_endpoint_config
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self):
        """Clean-up the resources associated with the Executor.

        GCE.shutdown will cancel all futures that have not yet registered with
        Globus Compute and will not wait for the launched futures to complete.
        """
        return self._executor.shutdown(wait=False, cancel_futures=True)
