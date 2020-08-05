from MC_RL_lib import *

trajs=50000
epsilon=0.15
yuanyang=YuanYangEnv()
train_EpsilonGonPolicy=MC_RL(yuanyang)
train_EpsilonGonPolicy.MC_RL_OnPolicy(trajs,epsilon)
yuanyang.Qtable=train_EpsilonGonPolicy.Qvalue
#画出MC_epsilon On Policy 方法的路径
flag=1
path=[]
step_num=0
#s=yuanyang.reset()
s=0
while flag==1:
    path.append(s)
    yuanyang.path=path
    a=train_EpsilonGonPolicy.epsilon_greedy_policy(train_EpsilonGonPolicy.Qvalue,s,epsilon)
    yuanyang.bird_male_position=yuanyang.state_to_position(s)
    yuanyang.render()
    step_num+=1
    next_s,r,flag=yuanyang.transform(s,a) 
    s=next_s
    pass
path.append(s)
yuanyang.bird_male_position=yuanyang.state_to_position(s)
yuanyang.render()
while True:
    yuanyang.render()
pass