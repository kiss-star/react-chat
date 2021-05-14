#------------------------------------------------------
# flatbuffers
#
# This mk file defines the flatbuffers-module with the prebuilt static library.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

FLATBUF_VER := @FLATBUF_VER@
ifeq ($(FLATBUF_VER),@FLATBUF_VER@)
$(error 'FLATBUF_VER' is not properly set)
endif

ifeq ($(shell which flatc),)
$(error No 'flatc' in your PATH, install flatbuffers-compiler from ppa:nnstreamer/ppa)
else
SYS_FLATC_VER := $(word 3, $(shell flatc --version))
endif

ifneq ($(SYS_FLATC_VER), $(FLATBUF_VER))
$(warning  Found 'flatc' v$(SYS_FLATC_VER), but required v$(FLATBUF_VER))
endif

FLATBUF_DIR := $(LOCAL_PATH)/flatbuffers
FLATBUF_INCLUDES := $(FLATBUF_DIR)/include
GEN_FLATBUF_HEADER := $(shell flatc --cpp -o $(LOCAL_PATH) $(NNSTREAMER_ROOT)/ext/nnstreamer/include/nnstreamer.fbs )
FLATBUF_HEADER_GEN := $(wildcard 