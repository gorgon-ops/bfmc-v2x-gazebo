# The environment server's verdict on one ObstacleReport.
std_msgs/Header header

uint16 vehicle_id
uint32 report_seq

bool accepted
float32 error      # m from the true obstacle, or -1 if nothing matched
string note        # human-readable reason, for the logs
