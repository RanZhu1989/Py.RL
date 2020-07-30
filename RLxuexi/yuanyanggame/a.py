from yuanyang import YuanYangEnv
import random
yuanyang=YuanYangEnv()
s=111
flag=yuanyang.collide(yuanyang.state_to_position(s))
print(yuanyang.state_to_position(s))
print(flag)