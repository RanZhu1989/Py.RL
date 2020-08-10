import random
import numpy as np
from Gridworld_wind import *

class Sarsa_Gridworld:
    #epsilon贪婪策略下的同轨TD(0)-Sarsa
    def __init__(self,Gridworld):
        self.Gridworld=Gridworld
        self.states=self.Gridworld.states
        self.actions=self.Gridworld.actions
        self.Qvalue=np.zeros((len(self.states),len(self.actions)))*0.1
        pass
    def action_to_num(self, a):
    # 返回动作的代号
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i
            pass
        pass
    def epsilon_greedy_policy(self, q, s, epsilon):
        # 定义epsilon-贪婪策略
        a_max = q[s, :].argmax()
        if np.random.uniform() < 1-epsilon:  # np包下random子库 uniform(low,high,size)产生low起始，high结束，size形状均匀分布
            return self.actions[a_max]
        else:
            return self.actions[int(random.random()*len(self.actions))]
        pass
    
    def sarsa_method(self, traj_num,alpha,epsilon):
        #Sarsa方法
        #Q_new(s,a) <- Q_old(s,a) + alpha *( R(s,a) + gamma * Q_old(s',a') - Q_old(s.a) )
        #a由pi(s):epsilon贪婪策略获得
        self.Qvalue=np.zeros((len(self.states),len(self.actions)))
        self.success_traj=[]
        for traj in range(traj_num):
            epsilon=epsilon*0.99
            #s=30#固定起点
            s = self.Gridworld.rest()
            a=self.epsilon_greedy_policy(self.Qvalue,s,epsilon)
            flag=0
            step_num=0
            while flag==0 and step_num<30:
                s_next,r,flag=self.Gridworld.transform(s,a)
                step_num+=1
                a_next = self.epsilon_greedy_policy(self.Qvalue,s_next,epsilon)
                a_num=self.action_to_num(a)
                a_next_num=self.action_to_num(a_next)
                if flag==1:
                    Qtarget=r
                    pass
                else:
                    Qtarget=r+self.Gridworld.gamma*self.Qvalue[s_next,a_next_num]
                    pass
                self.Qvalue[s,a_num]=self.Qvalue[s,a_num]+alpha*(Qtarget-self.Qvalue[s,a_num])
               
                s=s_next
                a=self.epsilon_greedy_policy(self.Qvalue,s,epsilon)
                if self.Gridworld.find(self.Gridworld.state_to_position(s))==1:
                    self.success_traj.append(traj_num)
                    pass
                pass
            pass
        return self.Qvalue
    
if __name__=="__main__":
    traj_num=10000
    epsilon=0.1
    alpha=0.5
    grid=GridWindEnv()
    train_sarsa=Sarsa_Gridworld(grid)
    train_sarsa.sarsa_method(traj_num,alpha,epsilon)
    print(train_sarsa.Qvalue)
    grid.Qtable=train_sarsa.Qvalue
    flag=0
    path=[]
    step_num=0
    s=30
    #s=grid.rest()
    while flag==0 and step_num<30:
        grid.plane_position=grid.state_to_position(s)
        a=train_sarsa.epsilon_greedy_policy(grid.Qtable,s,epsilon)
        s_next,r,flag=grid.transform(s,a)
        step_num+=1
        path.append(s)
        grid.path=path
        grid.render()
        s=s_next
        pass
    grid.path.append(s)
    print(grid.path)
    print(len(train_sarsa.success_traj))
    while True:
        grid.render()
        pass
   
    pass
pass