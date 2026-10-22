# Why the vehicle is doing what it is doing. Drives the HUD and the logs.
std_msgs/Header header

float32 commanded_speed
float32 steering_angle

string limiting_factor    # cruise | corner | signal | v2v
string reason             # human-readable justification

float32 arc_s             # distance along the route
float32 lateral_error     # m, signed, positive to the left
float32 distance_to_line  # m to the next stop line

string approach_decision  # PROCEED | PREPARE | STOP | CLEAR
string right_of_way       # PROCEED | YIELD | EMERGENCY_STOP
int32 conflicting_peer    # vehicle id, or -1
