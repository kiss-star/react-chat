
nns_capi_common_srcs = files('ml-api-common.c', 'ml-api-inference-internal.c')
if get_option('enable-tizen')
  if get_option('enable-tizen-feature-check')
    nns_capi_common_srcs += files('ml-api-common-tizen-feature-check.c')
  endif

  if get_option('enable-tizen-privilege-check')
    nns_capi_common_srcs += files('ml-api-inference-tizen-privilege-check.c')
  endif
endif

nns_capi_single_srcs = files('ml-api-inference-single.c')
nns_capi_pipeline_srcs = files('ml-api-inference-pipeline.c')
nns_capi_service_srcs = files('ml-api-service-common.c','ml-api-service-agent-client.c', 'ml-api-service-query-client.c')

# Build ML-API Common Lib First.
nns_capi_common_shared_lib = shared_library ('capi-ml-common',
  nns_capi_common_srcs,
  dependencies: nns_capi_common_deps,
  include_directories: nns_capi_include,
  install: true,
  install_dir: api_install_libdir,
  version: api_version,
)
nns_capi_common_static_lib = static_library ('capi-ml-common',
  nns_capi_common_srcs,
  dependencies: nns_capi_common_deps,