project('ml-api', 'c', 'cpp',
  version: '1.8.3',
  license: ['Apache-2.0'],
  meson_version: '>=0.50.0',
  default_options: [
    'b_asneeded=false',
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++14'
  ]
)

add_project_link_arguments('-Wl,--no-as-needed', language: 'c')
add_project_link_arguments('-Wl,--no-as-needed', language: 'cpp')

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

# Dependencies
glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
gmodule_dep = dependency('gmodule-2.0')
if host_machine.system() == 'windows'
  gio_dep = dependency('gio-2.0')
else
  gio_dep = [dependency('gio-2.0'), dependency('gio-unix-2.0')]
endif
gst_dep = dependency('gstreamer-1.0')
gst_app_dep = dependency('gstreamer-app-1.0')
nnstreamer_internal_dep = dependency('nnstreamer-internal')
nnstreamer_single_dep = dependency('nnstreamer-single')
nnstreamer_dep = dependency('nnstreamer')

if get_option('enable-ml-service')
  libsystemd_dep = dependency('libsystemd')
  sqlite_dep = dependency('sqlite3')
  json_glib_dep = dependency('json-glib-1.0')

  if get_option('enable-tizen')
    appfw_package_manager_dep = dependency('capi-appfw-package-manager')
  endif
endif

# Set ver