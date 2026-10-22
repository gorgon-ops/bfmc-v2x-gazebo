cmake_minimum_required(VERSION 3.8)
project(bfmc_v2x_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseFix.msg"
  "msg/SemaphoreState.msg"
  "msg/PeerState.msg"
  "msg/ObstacleReport.msg"
  "msg/ObstacleAck.msg"
  "msg/LocalizationDiagnostics.msg"
  "msg/DriveStatus.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
