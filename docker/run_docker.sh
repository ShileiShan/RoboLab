#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ROBOLAB_DIR=$( cd $( dirname ${BASH_SOURCE[0]} )/.. && pwd )

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
DEFAULT_TAG="$(docker images --format '{{.Repository}}:{{.Tag}}' | awk -F: -v repo="${IMAGE_NAME}" '$1==repo {print $2; exit}')"
DEFAULT_TAG="${DEFAULT_TAG:-$(git rev-parse --short HEAD)}"
IMAGE_TAG="${1:-$DEFAULT_TAG}"

xhost +local:root
docker run \
  -it \
  --entrypoint /bin/bash \
  -e DISPLAY \
  --net host \
  --rm \
  --runtime nvidia \
  -v $ROBOLAB_DIR/.cache/ov:/root/.cache/ov \
  -v $ROBOLAB_DIR/.cache/kit:/isaac-sim/kit/cache \
  -v $ROBOLAB_DIR:/workspace/robolab \
  -w /workspace/robolab \
  "${IMAGE_NAME}:${IMAGE_TAG}"
xhost -local:root
