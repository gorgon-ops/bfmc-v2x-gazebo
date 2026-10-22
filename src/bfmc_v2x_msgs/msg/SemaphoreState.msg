# One traffic light's colour and how long it will hold it.
uint8 RED=0
uint8 YELLOW=1
uint8 GREEN=2
uint8 FLASHING_YELLOW=3

std_msgs/Header header

uint16 light_id
uint8 state

# Seconds until the colour changes. Negative means the countdown could not
# be trusted -- the light has gone quiet, or its countdown already expired
# without a follow-up. Consumers must treat that as "colour only".
float32 ttl
