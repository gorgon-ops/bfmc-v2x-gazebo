cmake_minimum_required(VERSION 3.8)
project(bfmc_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY track urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/generate_world.py
  DESTINATION lib/${PROJECT_NAME}
)

# Point Gazebo at this package's assets when the workspace is sourced.
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/resource_paths.dsv.in")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Also verifies that worlds/mini_city.sdf is not stale with respect to
  # track/mini_city.yaml, which is the invariant this package exists for.
  ament_add_pytest_test(world_generation test)
endif()

ament_package()
