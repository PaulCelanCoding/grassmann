# Stub pointops2 -- the real CUDA extension fails to build inside the
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel image because cusolverDn.h is
# missing (the image ships cuda-runtime libraries but not full cuSOLVER
# headers, which torch's CUDAContext.h transitively requires). Yang's repo
# only uses pointops2 for the rigid-loss kNN; we run with lambda_rigid=0 in
# slice_banana.yaml so the actual functions are never called.
