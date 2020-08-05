from MC_RL_lib import *

traj_number_es=10000
yuanyang=YuanYangEnv()
train_ES=MC_RL(yuanyang)
train_ES.MC_learning_ES(traj_number_es)
yuanyang.Qtable=train_ES.Qvalue
#画出MC_ES方法的路径
flag=1
path=[]
step_num=0
s=yuanyang.reset()
while flag==1:
    path.append(s)
    yuanyang.path=path
    a=train_ES.greedy_policy(train_ES.Qvalue,s)
    yuanyang.bird_male_position=yuanyang.state_to_position(s)
    yuanyang.render()
    step_num+=1
    next_s,r,t=yuanyang.transform(s,a) 
    s=next_s
    if t==1 or step_num>30:
        flag=0
        pass
    pass
path.append(s)
yuanyang.bird_male_position=yuanyang.state_to_position(s)
yuanyang.render()
while True:
    yuanyang.render()
pass