LOCAL_PATH := $(call my-dir)

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

ifndef ML_API_ROOT
$(error ML_API_ROOT is not defined!)
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/armv7
else ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/arm64
else ifeq ($(TARGET_ARCH_ABI),x86)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/x86
else ifeq ($(TARGET_ARCH_ABI),x86_64)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/x86_64
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

# Set ML API Version
ML_API_VERSION  := 1.8.3
ML_API_VERSION_MAJOR := $(word 1,$(subst ., ,${ML_API_VERSION}))
ML_API_VERSION_MINOR := $(word 2,$(subst ., ,${ML_API_VERSION}))
ML_API_VERSION_MICRO := $(word 3,$(subst ., ,${ML_API_VERSION}))

#------------------------------------------------------
# API build option
#------------------------------------------------------
include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

NNSTREAMER_API_OPTION := all

# tensor-query support
ENABLE_TENSOR_QUERY := true

# tensorflow-lite (nnstreamer tf-lite subplugin)
ENABLE_TF_LITE := false

# SNAP (Samsung Neural Acceleration Platform)
ENABLE_SNAP := false

# NNFW (On-device neural network inference framework, Samsung Research)
ENABLE_NNFW := false

# SNPE (Snapdragon Neural Processing Engine)
ENABLE_SNPE := false

# PyTorch
ENABLE_PYTORCH := false

# MXNet
ENABLE_MXNET := false

# C