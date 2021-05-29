#------------------------------------------------------
# SNPE (The Snapdragon Neural Processing Engine)
#
# This mk file defines snpe module with prebuilt shared library.
# (snpe-sdk, arm64-v8a only)
# See Qualcomm Neural Processing SDK for AI (https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) for the details.
#
# You should check your `gradle.properties` to set the variable `SNPE_EXT_LIBRARY_PATH` properly.
# The variable should be assigend with path for external shared libs.
# An example: "SNPE_EXT_LIBRARY_PATH=src/main/jni/snpe/lib/ext"
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

SNPE_DIR := $(LOCAL_PATH)/snpe
SNPE_IN