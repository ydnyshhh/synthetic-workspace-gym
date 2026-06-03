from __future__ import annotations


class SandboxError(RuntimeError):
    pass


class DockerUnavailableError(SandboxError):
    pass


class SandboxTimeoutError(SandboxError):
    pass


class SandboxPolicyError(SandboxError):
    pass


class SandboxExecutionError(SandboxError):
    pass
