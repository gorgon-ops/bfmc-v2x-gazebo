# An absolute pose from the trackside localisation system.
#
# header.stamp is when this message was published; measured_at is when the
# camera frame it came from was exposed. They differ by the tracker's
# processing time plus the network delay, and the difference is exactly what
# the filter's delayed-fusion path exists to account for -- so consumers
# must fuse against measured_at, never against header.stamp.
std_msgs/Header header
builtin_interfaces/Time measured_at

uint16 vehicle_id

float32 x          # m, track frame
float32 y          # m, track frame
float32 yaw        # rad, counter-clockwise from +x

float32 sigma_pos  # m, 1-sigma, as reported by the tracker
float32 sigma_yaw  # rad, 1-sigma
