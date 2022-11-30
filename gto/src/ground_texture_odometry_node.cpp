#include <gto/ground_texture_odometry.h>

int main(int argc, char **argv){
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  ros::init(argc, argv, "ground_texture_odometry_node");

  ground_texture_odometry::GroundTextureOdometry* gto = new ground_texture_odometry::GroundTextureOdometry();
  gto->main();
  
  delete gto;
  return 0;
}
