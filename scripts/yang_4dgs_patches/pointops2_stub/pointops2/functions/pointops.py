"""Stub for pointops2.functions.pointops -- imports only succeed; calling
either function raises so we fail fast if our config ever flips on rigid loss."""


def _stub(*args, **kwargs):
    raise NotImplementedError(
        "pointops2 was not built into the Yang-4DGS image (cusolverDn.h missing "
        "in cuda-11.6 base image). This stub is only safe when lambda_rigid=0."
    )


furthestsampling = _stub
knnquery = _stub
