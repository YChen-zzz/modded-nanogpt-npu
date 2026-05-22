# Runtime image used for this Ascend NPU port.
#
# This Dockerfile records the container image and Python environment used during
# validation. It is not the original CUDA/H100 Dockerfile from upstream.

FROM docker.cnb.cool/nilpotenter/docker/codeserver-mindspeed:v1.0.5

ENV CONDA_PREFIX=/root/miniconda3/envs/llm_test
ENV PATH="${CONDA_PREFIX}/bin:${PATH}"

WORKDIR /workspace/modded-nanogpt-record50-cautious-wd

CMD ["bash"]
ENTRYPOINT []
