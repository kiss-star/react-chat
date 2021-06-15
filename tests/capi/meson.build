unittest_util_static = static_library('unittest_util',
  files('unittest_util.c'),
  dependencies: [glib_dep],
  install: false,
)

unittest_capi_inference_single = executable('unittest_capi_inference_single',
  'unittest_capi_inference_single.cc',
  dependencies: [nns_capi_single_dep, gtest_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir,
  include_directories: nns_capi_include,
)
test('unittest_capi_inference_single', unittest_c