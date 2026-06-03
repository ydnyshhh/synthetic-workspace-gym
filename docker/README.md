# SWG Docker Runtime

Build the runtime image:

```bash
docker build -f docker/Dockerfile.swg-runtime -t synthetic-workspace-gym-runtime:latest .
```

Tool mode mounts only the visible workspace at `/workspace` and disables network by default. Hidden evaluator assets are not mounted for model-facing tools.

Evaluator mode mounts the visible workspace at `/workspace` and hidden assets at `/hidden` read-only for trusted verification.

This container runtime is a stronger isolation layer than local subprocess execution, but it is not a perfect sandbox for hostile code.
