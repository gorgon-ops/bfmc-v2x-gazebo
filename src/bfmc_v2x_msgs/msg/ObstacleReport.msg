# Something a vehicle detected, placed on the map with its own pose estimate.
uint8 STATIC_CAR=0
uint8 PEDESTRIAN=1
uint8 ROADBLOCK=2
uint8 STOP_SIGN=3
uint8 CROSSWALK=4
uint8 PARKING_SPOT=5

std_msgs/Header header

uint16 vehicle_id
uint32 seq

uint8 kind
float32 x
float32 y
float32 confidence
