/**
 * @file        unittest_capi_service_agent_client.cc
 * @date        20 Jul 2022
 * @brief       Unit test for ML Service C-API.
 * @see         https://github.com/nnstreamer/api
 * @author      Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug         No known bugs
 */

#include <gio/gio.h>
#include <gtest/gtest.h>
#include <ml-api-internal.h>
#include <ml-api-service.h>
#include <ml-api-service-private.h>
#include <ml-api-inference-pipeline-internal