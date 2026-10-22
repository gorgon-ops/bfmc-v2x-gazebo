# Cooperative awareness message from another vehicle.
uint8 STRAIGHT=0
uint8 LEFT=1
uint8 RIGHT=2
uint8 HOLD=3

std_msgs/Header header

uint16 vehicle_id

float32 x
float32 y
float32 yaw
float32 speed
float32 yaw_rate

uint8 intent
float32 age        # s since the peer stamped it, network delay included
