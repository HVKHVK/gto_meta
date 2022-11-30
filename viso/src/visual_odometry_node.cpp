#include <viso/visual_odometry.h>

int main(int argc, char **argv){
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  ros::init(argc, argv, "visual_odometry_node");

  visual_odometry::VisualOdometry* viso = new visual_odometry::VisualOdometry();
  viso->main();
  
  delete viso;
  return 0;
}
