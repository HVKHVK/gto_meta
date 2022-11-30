cmake_minimum_required(VERSION 2.8.3)
project(gto)
add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)

set(DEPS
  roscpp
  sensor_msgs
  cv_bridge
  std_msgs
  image_transport
  dynamic_reconfigure
  message_generation
  nav_msgs
  tf2_ros
  geometry_msgs
)

find_package(catkin REQUIRED COMPONENTS ${DEPS})
find_package(OpenCV REQUIRED )

generate_dynamic_reconfigure_options(
   cfg/gto.cfg
)

add_service_files(
  FILES
  PoseCmd.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS ${DEPS}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(ground_texture_odometry src/ground_texture_odometry.cpp)
add_dependencies(ground_texture_odometry ${PROJECT_NAME}_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ground_texture_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(ground_texture_odometry_node src/ground_texture_odometry_node.cpp)
add_dependencies(ground_texture_odometry_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ground_texture_odometry_node ground_texture_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

set_target_properties(ground_texture_odometry_node PROPERTIES OUTPUT_NAME ground_texture_odometry_node)

install(TARGETS ground_texture_odometry_node ground_texture_odometry 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)
