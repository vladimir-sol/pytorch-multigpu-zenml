"""Settings for the remote ZenML stack.

Docker and Skypilot settings are ignored on the local (default) stack, since
the local venv is used instead of Docker and local GPU is used instead of a
Skypilot VM.
"""

from zenml.config import DockerSettings
from zenml.config.docker_settings import (
    # PythonPackageInstaller,
    PythonEnvironmentExportMethod,
)
from zenml.integrations.skypilot_aws.flavors.skypilot_orchestrator_aws_vm_flavor import (
    SkypilotAWSOrchestratorSettings,
)


docker = DockerSettings(
    # # Fsspec resolution conflict: pip resolves fsspec to a yanked version and
    # # fails. As a workaround, either use UV or export stack requirements and
    # # install them as part of user requirements.
    # python_package_installer=PythonPackageInstaller.UV,

    replicate_local_python_environment=PythonEnvironmentExportMethod.POETRY_EXPORT,
)


skypilot = SkypilotAWSOrchestratorSettings(
    region="ca-central-1",
    instance_type="g4dn.xlarge",
    # instance_type="g4dn.12xlarge",  # T4:4
    # num_nodes=4,  # supported in a future release https://github.com/zenml-io/zenml/pull/3612
    use_spot=True,
    synchronous=False,  # undocumented and might not work
    docker_run_args=[
        "--gpus=all",
        "--ipc=host",  # for single node multi-gpu setups; security caveats
    ],
)
