
#!/usr/bin/env bash

##
## SPDX-License-Identifier: Apache-2.0
##
# @file  build-nnstreamer-android.sh
# @brief A script to build NNStreamer API library for Android
#
# The following comments that start with '##@@' are for the generation of usage messages.
##@@ Build script for Android NNStreamer API Library
##@@  - Before running this script, below variables must be set.
##@@  - ANDROID_SDK_ROOT: Android SDK
##@@  - ANDROID_NDK_ROOT: Android NDK
##@@  - GSTREAMER_ROOT_ANDROID: GStreamer prebuilt libraries for Android
##@@  - NNSTREAMER_ROOT: The source root directory of NNStreamer
##@@  - NNSTREAMER_EDGE_ROOT: The source root directory of nnstreamer-edge
##@@  - ML_API_ROOT: The source root directory of ML API
##@@ 
##@@ usage: build-nnstreamer-android.sh [OPTIONS]
##@@ 
##@@ basic options:
##@@   --help
##@@       display this help and exit
##@@   --build-type=(all|lite|single|internal)
##@@       'all'      : default
##@@       'lite'     : build with GStreamer core plugins
##@@       'single'   : no plugins, single-shot only
##@@       'internal' : no plugins except for enable single-shot only, enable NNFW only
##@@   --target_abi=(armeabi-v7a|arm64-v8a)
##@@       'arm64-v8a' is the default Android ABI
##@@   --run_test=(yes|no)
##@@       'yes'      : run instrumentation test after build procedure is done
##@@       'no'       : [default]
##@@   --nnstreamer_dir=(the_source_root_of_nnstreamer)
##@@       This option overrides the NNSTREAMER_ROOT variable
##@@   --ml_api_dir=(the_source_root_of_ml_api)
##@@       This option overrides the ML_API_ROOT variable
##@@   --result_dir=(path_to_build_result)
##@@       Default path is 'ml_api_dir/android_lib'
##@@ 
##@@ options for GStreamer build:
##@@   --enable_tracing=(yes|no)
##@@       'yes'      : build with GStreamer Tracing feature
##@@       'no'       : [default]
##@@ 
##@@ options for tensor filter sub-plugins:
##@@   --enable_snap=(yes|no)
##@@       'yes'      : build with sub-plugin for SNAP
##@@                    This option requires 1n additional variable, 'SNAP_DIRECTORY',
##@@                    which indicates the SNAP SDK interface's absolute path.
##@@       'no'       : [default]
##@@   --enable_nnfw=(yes|no)
##@@       'yes'      : [default]
##@@       'no'       : build without the sub-plugin for NNFW
##@@   --enable_snpe=(yes|no)
##@@       'yes'      : build with sub-plugin for SNPE
##@@       'no'       : [default]
##@@   --enable_pytorch=(yes(:(1.10.1))?|no)
##@@       'yes'      : build with sub-plugin for PyTorch. You can optionally specify the version of
##@@                    PyTorch to use by appending ':version' [1.10.1 is the default].
##@@       'no'       : [default] build without the sub-plugin for PyTorch
##@@   --enable_tflite=(yes(:(1.9|1.13.1|1.15.2|2.3.0|2.7.0|2.8.1))?|no)
##@@       'yes'      : [default] you can optionally specify the version of tensorflow-lite to use
##@@                    by appending ':version' [2.8.1 is the default].
##@@       'no'       : build without the sub-plugin for tensorflow-lite
##@@   --enable_mxnet=(yes|no)
##@@       'yes'      : build with sub-plugin for MXNet. Currently, mxnet 1.9.1 version supported.
##@@       'no'       : [default] build without the sub-plugin for MXNet
##@@ 
##@@ options for tensor converter/decoder sub-plugins:
##@@   --enable_flatbuf=(yes|no)
##@@       'yes'      : [default]
##@@       'no'       : build without the sub-plugin for FlatBuffers and FlexBuffers
##@@ 
##@@ options for tensor_query:
##@@   --enable_tensor_query=(yes|no)
##@@       'yes'      : [default] build with tensor_query elements.
##@@       'no'       : build without the tensor_query elements.
##@@
##@@ options for mqtt:
##@@   --enable_mqtt=(yes|no)
##@@       'yes'      : [default] build with paho.mqtt.c prebuilt libs. This option supports the mqtt plugin
##@@       'no'       : build without the mqtt support
##@@
##@@ For example, to build library with core plugins for arm64-v8a
##@@  ./build-nnstreamer-android.sh --api_option=lite --target_abi=arm64-v8a

# API build option
# 'all' : default
# 'lite' : with GStreamer core plugins
# 'single' : no plugins, single-shot only
# 'internal' : no plugins, single-shot only, enable NNFW only
build_type="all"

nnstreamer_api_option="all"
include_assets="no"

# Set target ABI ('armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64')
target_abi="arm64-v8a"

# Run instrumentation test after build procedure is done
run_test="no"

# Enable GStreamer Tracing
enable_tracing="no"

# Enable SNAP
enable_snap="no"

# Enable NNFW
enable_nnfw="yes"

# Enable SNPE
enable_snpe="no"

# Enable PyTorch
enable_pytorch="no"

# Set PyTorch version (available: 1.8.0 (unstable) / 1.10.1)
pytorch_ver="1.10.1"
pytorch_vers_support="1.8.0 1.10.1"

# Enable tensorflow-lite
enable_tflite="yes"

# Enable MXNet
enable_mxnet="no"
mxnet_ver="1.9.1"

# Enable the flatbuffer converter/decoder by default
enable_flatbuf="no"
flatbuf_ver="1.12.0"

# Enable option for MQTT
enable_mqtt="no"
paho_mqtt_c_ver="1.3.7"

# Enable option for tensor_query
enable_tensor_query="yes"

# Set tensorflow-lite version (available: 1.9.0 / 1.13.1 / 1.15.2 / 2.3.0 / 2.7.0 / 2.8.1)
tf_lite_ver="2.8.1"
tf_lite_vers_support="1.9.0 1.13.1 1.15.2 2.3.0 2.7.0 2.8.1"

# Set NNFW version (https://github.com/Samsung/ONE/releases)
nnfw_ver="1.17.0"
enable_nnfw_ext="no"

# Find '--help' in the given arguments
arg_help="--help"
for arg in "$@"; do
    if [[ $arg == $arg_help ]]; then
        sed -ne 's/^##@@ \(.*\)/\1/p' $0 && exit 1
    fi
done

# Parse args
for arg in "$@"; do
    case $arg in
        --build_type=*)
            build_type=${arg#*=}
            ;;
        --target_abi=*)
            target_abi=${arg#*=}
            ;;
        --run_test=*)
            run_test=${arg#*=}
            ;;
        --nnstreamer_dir=*)
            nnstreamer_dir=${arg#*=}
            ;;
        --nnstreamer_edge_dir=*)
            nnstreamer_edge_dir=${arg#*=}
            ;;
        --ml_api_dir=*)
            ml_api_dir=${arg#*=}
            ;;
        --result_dir=*)
            result_dir=${arg#*=}
            ;;
        --gstreamer_dir=*)
            gstreamer_dir=${arg#*=}
            ;;
        --android_sdk_dir=*)
            android_sdk_dir=${arg#*=}
            ;;
        --android_ndk_dir=*)
            android_ndk_dir=${arg#*=}
            ;;
        --enable_tracing=*)
            enable_tracing=${arg#*=}
            ;;
        --enable_snap=*)
            enable_snap=${arg#*=}
            ;;
        --enable_nnfw=*)
            enable_nnfw=${arg#*=}
            ;;
        --enable_nnfw_ext=*)
            enable_nnfw_ext=${arg#*=}
            ;;
        --enable_snpe=*)
            enable_snpe=${arg#*=}
            ;;
        --enable_pytorch=*)
            IFS=':' read -ra enable_pytorch_args <<< "${arg#*=}"
            is_valid_pytorch_version=0
            enable_pytorch=${enable_pytorch_args[0]}
            if [[ ${enable_pytorch} == "yes" ]]; then
                if [[ ${enable_pytorch_args[1]} == "" ]]; then
                    break
                fi
                for ver in ${pytorch_vers_support}; do
                    if [[ ${ver} == ${enable_pytorch_args[1]} ]]; then
                        is_valid_pytorch_version=1
                        pytorch_ver=${ver}
                        break
                    fi
                done
                if [[ ${is_valid_pytorch_version} == 0 ]]; then
                    printf "'%s' is not a supported version of PyTorch." "${enable_pytorch_args[1]}"
                    printf "The default version, '%s', will be used.\n"  "${pytorch_ver}"
                fi
            fi
            ;;
        --enable_tflite=*)
            IFS=':' read -ra enable_tflite_args <<< "${arg#*=}"
            is_valid_tflite_version=0
            enable_tflite=${enable_tflite_args[0]}
            if [[ ${enable_tflite} == "yes" ]]; then
                if [[ ${enable_tflite_args[1]} == "" ]]; then
                    break
                fi
                for ver in ${tf_lite_vers_support}; do
                    if [[ ${ver} == ${enable_tflite_args[1]} ]]; then
                        is_valid_tflite_version=1
                        tf_lite_ver=${ver}
                        break