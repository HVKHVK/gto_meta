cmake_minimum_required(VERSION 2.8.3)
project(viso)
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
   cfg/viso.cfg
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

add_library(visual_odometry src/visual_odometry.cpp)
add_dependencies(visual_odometry ${PROJECT_NAME}_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(visual_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(visual_odometry_node src/visual_odometry_node.cpp)
add_dependencies(visual_odometry_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(visual_odometry_node visual_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

set_target_properties(visual_odometry_node PROPERTIES OUTPUT_NAME visual_odometry_node)

install(TARGETS visual_odometry_node visual_odometry 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
