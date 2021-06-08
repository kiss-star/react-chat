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

cc = me