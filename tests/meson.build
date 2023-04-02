# Install path for unittest
unittest_base_dir = join_paths(api_install_bindir, 'unittest-ml')
unittest_install_dir = join_paths(unittest_base_dir,'tests')

# Set dependency and test-env
testenv = environment()
testenv.set('MLAPI_SOURCE_ROOT_PATH', meson.source_root())
testenv.set('MLAPI_BUILD_ROOT_PATH', meson.build_root())

gtest_dep = dependency('gtest', required: false)
if gtest_dep.found()
  if get_option('enable-ml-service')
    gdbus_gen_test_src = custom_target('gdbus-gencode-test',
      input: test_dbus_input,
      output: ['test-dbus.h', 'test-dbus.c'],
      command: [gdbus_prog, '--interface-prefix', 'org.tizen',
                '--generate-c-code', 'test-dbus',
