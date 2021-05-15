#------------------------------------------------------
# Define GStreamer plugins and extra dependencies
#------------------------------------------------------

ifndef NNSTREAMER_API_OPTION
$(error NNSTREAMER_API_OPTION is not defined!)
endif

ifndef GSTREAMER_NDK_BUILD_PATH
GSTREAMER_NDK_BUILD_PATH := $(GSTREAMER_ROOT)/share/gst-android/ndk-build
endif

include $(GSTREAMER_NDK_BUILD_PATH)/plugins.mk

ifeq ($(NNSTREAMER_API_OPTION),all)
GST_REQUIRED_PLUGINS := $(GSTREAMER_PLUGINS_CORE) \
    