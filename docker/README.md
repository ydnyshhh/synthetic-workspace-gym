# SWG Docker Runtime

Build the runtime image:

```bash
docker build -f docker/Dockerfile.swg-runtime -t synthetic-workspace-gym-runtime:latest .
```

Tool mode mounts only the visible workspace at `/workspace` and disables network by default. Hidden evaluator assets are not mounted for model-facing tools. Tool containers receive a minimal runtime environment instead of the host environment, so local credentials and API keys are not forwarded by default.

Evaluator mode mounts the visible workspace at `/workspace` and hidden assets at `/hidden` read-only for trusted verification.

On POSIX hosts the CLI defaults Docker `--user` to the current uid/gid so bind mounts stay writable. Pass `--sandbox-user UID:GID` when a different mapping is needed.

This container runtime is a stronger isolation layer than local subprocess execution, but it is not a perfect sandbox for hostile code.
