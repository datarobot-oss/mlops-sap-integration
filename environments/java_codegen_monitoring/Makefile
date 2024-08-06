SHELL = /bin/sh

NS := datarobot-oss
IMAGE_NAME := mlops-sap-monitoring-scoring-code
VERSION := 1.0.0

REGISTRY := ghcr.io
DOCKERFILE := Dockerfile

ifeq ($(REGISTRY),)
  fq_image := $(NS)/$(IMAGE_NAME)
else
  fq_image := $(REGISTRY)/$(NS)/$(IMAGE_NAME)
endif

MAJOR_VERSION = $(shell echo $(VERSION) | sed 's/\.[^\.]*$$//')

print:
	@echo "Building image for scoring code predictor: "
	@echo "	$(fq_image):$(VERSION)"
	@echo "	$(fq_image):$(MAJOR_VERSION)"
	@echo "	$(fq_image):latest"

build: ## > build - To build the this docker image
	docker build --pull \
		--cache-from $(fq_image):latest \
		-t $(fq_image):$(VERSION) \
		-t $(fq_image):$(MAJOR_VERSION) \
		-t $(fq_image):latest \
		-f $(DOCKERFILE) \
		.

push: build ## > push - Pushes this image to a remote registry. Use the NS and REGISTRY variables to control the name of the image that is pushed.
	docker push $(fq_image):$(MAJOR_VERSION)
	docker push $(fq_image):$(VERSION)
	docker push $(fq_image):latest

run:
	@echo "Starting scoring code predictor"
	docker run --rm -it \
	 -p 9001:9001 \
	 -e DATAROBOT_ENDPOINT \
	 -e DATAROBOT_API_TOKEN \
	 -e MLOPS_DEPLOYMENT_ID \
	 -e MLOPS_MODEL_ID \
	 -e MLOPS_MODEL_PACKAGE_ID \
	 -e TARGET_TYPE \
	 -e POSITIVE_CLASS_LABEL \
	 -e NEGATIVE_CLASS_LABEL \
	 -e CLASS_LABELS \
	 -e MONITOR \
	 -e MONITOR_SETTINGS \
	  $(IMAGE_NAME):$(IMAGE_TAG)
