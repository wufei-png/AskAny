# host device setting
uname_p := $(shell uname -p)
ifeq (${host},)
	ifeq (${uname_p}, x86_64)
		host := amd64
	else ifeq (${uname_p}, aarch64)
		host := arm64
	else
		host := amd64
	endif
endif

ifeq (${device},)
	ifneq (, $(shell ls /dev/nvidiactl 2>/dev/null))
		device := cuda11
	else ifneq (, $(shell ls /dev/hisi_hdc 2>/dev/null))
		device := atlas310p
	else ifneq (, $(shell ls /dev/stpu/stpuctrl 2>/dev/null))
		device := stpu
	else
		device := cuda11
	endif
endif
export host := ${host}
export device := ${device}

export nolic?=0
export cross_compile?=0
export host := ${host}
export GOARCH=${host}
export VERSION=v2.0.0

TAG := $(shell git describe --tags --exact-match 2>/dev/null)
ifeq ($(TAG),)
    TAG := ${VERSION}-dev-$(shell git rev-parse --short=8 HEAD)
endif
export TAG := $(subst +,-,$(TAG))

host_libs=${host}
device_libs=${host}_${device}
rc=${host}_${device}

ifeq ($(kestrel_version), v1)
	rc=${host}_${device}_kestrelV1
endif

ifeq ($(cross_compile), 1)
	rc := ${rc}_cross
endif

ifeq ($(shell test -e .makerc/${rc} && echo -n yes),yes)
	include .makerc/${rc}
endif

COMMIT_TAG := $(shell git rev-parse --short=8 HEAD)

DOCKER_MOUNT := ${PWD}:${code_root}
DEV_DOCKERFILE ?= Dockerfile.dev
DEV_IMAGE_NAME ?= askany-dev:v2.0

SRC = $(shell find . -type f -name '*.go' -not -path "./vendor/*")

.PHONY: build_dev_image

build_dev_image:
	docker buildx build --load -f $(DEV_DOCKERFILE) -t $(DEV_IMAGE_NAME) .