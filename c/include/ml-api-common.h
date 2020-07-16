
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer API / Tizen Machine-Learning API Common Header
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file    ml-api-common.h
 * @date    07 May 2020
 * @brief   ML-API Common Header
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * @details
 *      More entries might be migrated from nnstreamer.h if
 *    other modules of Tizen-ML APIs use common data structures.
 */
#ifndef __ML_API_COMMON_H__
#define __ML_API_COMMON_H__

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <tizen_error.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_FRAMEWORK
 * @{
 */

/**
 * @brief Enumeration for the error codes of NNStreamer.
 * @since_tizen 5.5
 */
typedef enum {
  ML_ERROR_NONE                 = TIZEN_ERROR_NONE, /**< Success! */
  ML_ERROR_INVALID_PARAMETER    = TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  ML_ERROR_STREAMS_PIPE         = TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create or access the pipeline. */
  ML_ERROR_TRY_AGAIN            = TIZEN_ERROR_TRY_AGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
  ML_ERROR_UNKNOWN              = TIZEN_ERROR_UNKNOWN,  /**< Unknown error */
  ML_ERROR_TIMED_OUT            = TIZEN_ERROR_TIMED_OUT,  /**< Time out */
  ML_ERROR_NOT_SUPPORTED        = TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  ML_ERROR_PERMISSION_DENIED    = TIZEN_ERROR_PERMISSION_DENIED, /**< Permission denied */
  ML_ERROR_OUT_OF_MEMORY        = TIZEN_ERROR_OUT_OF_MEMORY, /**< Out of memory (Since 6.0) */
  ML_ERROR_IO_ERROR             = TIZEN_ERROR_IO_ERROR, /**< I/O error for database and filesystem (Since 7.0) */
} ml_error_e;

/**
 * @brief Types of NNFWs.
 * @details To check if a nnfw-type is supported in a system, an application may call the API, ml_check_nnfw_availability().
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_TYPE_ANY = 0,               /**< NNFW is not specified (Try to determine the NNFW with file extension). */
  ML_NNFW_TYPE_CUSTOM_FILTER = 1,     /**< Custom filter (Independent shared object). */
  ML_NNFW_TYPE_TENSORFLOW_LITE = 2,   /**< Tensorflow-lite (.tflite). */
  ML_NNFW_TYPE_TENSORFLOW = 3,        /**< Tensorflow (.pb). */
  ML_NNFW_TYPE_NNFW = 4,              /**< Neural Network Inference framework, which is developed by SR (Samsung Research). */
  ML_NNFW_TYPE_MVNC = 5,              /**< Intel Movidius Neural Compute SDK (libmvnc). (Since 6.0) */
  ML_NNFW_TYPE_OPENVINO = 6,          /**< Intel OpenVINO. (Since 6.0) */
  ML_NNFW_TYPE_VIVANTE = 7,           /**< VeriSilicon's Vivante. (Since 6.0) */
  ML_NNFW_TYPE_EDGE_TPU = 8,          /**< Google Coral Edge TPU (USB). (Since 6.0) */
  ML_NNFW_TYPE_ARMNN = 9,             /**< Arm Neural Network framework (support for caffe and tensorflow-lite). (Since 6.0) */
  ML_NNFW_TYPE_SNPE = 10,             /**< Qualcomm SNPE (Snapdragon Neural Processing Engine (.dlc). (Since 6.0) */
  ML_NNFW_TYPE_PYTORCH = 11,          /**< PyTorch (.pt). (Since 6.5) */
  ML_NNFW_TYPE_NNTR_INF = 12,         /**< Inference supported from NNTrainer, SR On-device Training Framework (Since 6.5) */
  ML_NNFW_TYPE_VD_AIFW = 13,          /**< Inference framework for Samsung Tizen TV (Since 6.5) */
  ML_NNFW_TYPE_TRIX_ENGINE = 14,      /**< TRIxENGINE accesses TRIV/TRIA NPU low-level drivers directly (.tvn). (Since 6.5) You may need to use high-level drivers wrapping this low-level driver in some devices: e.g., AIFW */
  ML_NNFW_TYPE_MXNET = 15,            /**< Apache MXNet (Since 7.0) */
  ML_NNFW_TYPE_TVM = 16,              /**< Apache TVM (Since 7.0) */
  ML_NNFW_TYPE_SNAP = 0x2001,         /**< SNAP (Samsung Neural Acceleration Platform), only for Android. (Since 6.0) */
} ml_nnfw_type_e;

/**
 * @brief Types of hardware resources to be used for NNFWs. Note that if the affinity (nnn) is not supported by the driver or hardware, it is ignored.
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_HW_ANY          = 0,      /**< Hardware resource is not specified. */
  ML_NNFW_HW_AUTO         = 1,      /**< Try to schedule and optimize if possible. */
  ML_NNFW_HW_CPU          = 0x1000, /**< 0x1000: any CPU. 0x1nnn: CPU # nnn-1. */
  ML_NNFW_HW_CPU_SIMD     = 0x1100, /**< 0x1100: SIMD in CPU. (Since 6.0) */
  ML_NNFW_HW_CPU_NEON     = 0x1100, /**< 0x1100: NEON (alias for SIMD) in CPU. (Since 6.0) */
  ML_NNFW_HW_GPU          = 0x2000, /**< 0x2000: any GPU. 0x2nnn: GPU # nnn-1. */
  ML_NNFW_HW_NPU          = 0x3000, /**< 0x3000: any NPU. 0x3nnn: NPU # nnn-1. */
  ML_NNFW_HW_NPU_MOVIDIUS = 0x3001, /**< 0x3001: Intel Movidius Stick. (Since 6.0) */
  ML_NNFW_HW_NPU_EDGE_TPU = 0x3002, /**< 0x3002: Google Coral Edge TPU (USB). (Since 6.0) */
  ML_NNFW_HW_NPU_VIVANTE  = 0x3003, /**< 0x3003: VeriSilicon's Vivante. (Since 6.0) */
  ML_NNFW_HW_NPU_SLSI     = 0x3004, /**< 0x3004: Samsung S.LSI. (Since 6.5) */
  ML_NNFW_HW_NPU_SR       = 0x13000, /**< 0x13000: any SR (Samsung Research) made NPU. (Since 6.0) */
} ml_nnfw_hw_e;

/******* ML API Common Data Structure for Inference, Training, and Service */
/**
 * @brief The maximum rank that NNStreamer supports with Tizen APIs.
 * @remarks The maximum rank in Tizen APIs is 4 until tizen 7.0 and 16 since 7.5
 * @since_tizen 5.5
 */
#define ML_TENSOR_RANK_LIMIT  (16)

/**
 * @brief The maximum number of other/tensor instances that other/tensors may have.
 * @since_tizen 5.5
 */
#define ML_TENSOR_SIZE_LIMIT  (16)

/**
 * @brief The dimensions of a tensor that NNStreamer supports.
 * @since_tizen 5.5
 */
typedef unsigned int ml_tensor_dimension[ML_TENSOR_RANK_LIMIT];

/**
 * @brief A handle of a tensors metadata instance.
 * @since_tizen 5.5
 */
typedef void *ml_tensors_info_h;

/**
 * @brief A handle of input or output frames. #ml_tensors_info_h is the handle for tensors metadata.
 * @since_tizen 5.5
 */
typedef void *ml_tensors_data_h;

/**
 * @brief Possible data element types of tensor in NNStreamer.
 * @since_tizen 5.5
 */
typedef enum _ml_tensor_type_e
{
  ML_TENSOR_TYPE_INT32 = 0,      /**< Integer 32bit */
  ML_TENSOR_TYPE_UINT32,         /**< Unsigned integer 32bit */
  ML_TENSOR_TYPE_INT16,          /**< Integer 16bit */
  ML_TENSOR_TYPE_UINT16,         /**< Unsigned integer 16bit */
  ML_TENSOR_TYPE_INT8,           /**< Integer 8bit */
  ML_TENSOR_TYPE_UINT8,          /**< Unsigned integer 8bit */
  ML_TENSOR_TYPE_FLOAT64,        /**< Float 64bit */
  ML_TENSOR_TYPE_FLOAT32,        /**< Float 32bit */
  ML_TENSOR_TYPE_INT64,          /**< Integer 64bit */
  ML_TENSOR_TYPE_UINT64,         /**< Unsigned integer 64bit */
  ML_TENSOR_TYPE_FLOAT16,        /**< FP16, IEEE 754. Note that this type is supported only in aarch64/arm devices. (Since 7.0) */
  ML_TENSOR_TYPE_UNKNOWN         /**< Unknown type */
} ml_tensor_type_e;

/**
 * @brief The function to be called when destroying the data in machine learning API.
 * @since_tizen 7.0
 * @param[in] data The data to be destroyed.
 */
typedef void (*ml_data_destroy_cb) (void *data);

/**
 * @brief Callback to execute the custom-easy filter in NNStreamer pipelines.
 * @details Note that if ml_custom_easy_invoke_cb() returns negative error values, the constructed pipeline does not work properly anymore.
 *          So developers should release the pipeline handle and recreate it again.
 * @since_tizen 6.0
 * @remarks The @a in can be used only in the callback. To use outside, make a copy.
 * @remarks The @a out can be used only in the callback. To use outside, make a copy.
 * @param[in] in The handle of the tensor input (a single frame. tensor/tensors).
 * @param[out] out The handle of the tensor output to be filled (a single frame. tensor/tensors).
 * @param[in,out] user_data User application's private data.
 * @return @c 0 on success. @c 1 to ignore the input data. Otherwise a negative error value.
 */
typedef int (*ml_custom_easy_invoke_cb) (const ml_tensors_data_h in, ml_tensors_data_h out, void *user_data);

/****************************************************
 ** NNStreamer Utilities                           **
 ****************************************************/
/**
 * @brief Creates a tensors information handle with default value.
 * @since_tizen 5.5
 * @param[out] info The handle of tensors information.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_tensors_info_create (ml_tensors_info_h *info);

/**
 * @brief Creates an extended tensors information handle with default value.
 * @details An extended tensors support higher rank limit.
 * @since_tizen 7.5
 * @param[out] info The handle of tensors information.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.