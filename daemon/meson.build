# Machine Learning Agent
# This file should be included when ml-service feature is enabled.
nns_ml_agent_incs = include_directories('includes')
nns_ml_agent_common_srcs = files('main.c', 'modules.c', 'gdbus-util.c',
  'pipeline-dbus-impl.cc', 'model-dbus-impl.cc')
nns_ml_agent_service_db_srcs = files('service-db.cc')

if (get_option('enable-tizen'))
  nns_ml_agent_common_srcs += files('pkg-mgr.c')
endif

nns_ml_agent_srcs = [nns_ml_agent_common_srcs, nns_ml_agent_service_db_srcs]

# Generate GDbus header and code
gdbus_prog = find_program('gdbus-codegen', required: true)
gdbus_gen_pipeline_src = custom_target('gdbus-gencode',
  input: pipeline_dbus_input,
  output: ['pipeline-dbus.h', 'pipeline-dbus.c'],
  command: [gdbus_prog, '--interface-prefix', 'org.tizen',
            '--generate-c-code', 'pipeline-dbus',
