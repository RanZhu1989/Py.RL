import random
import numpy as np
import matplotlib.pyplot as plt
from Gridworld_wind import *

class Q_Gridworld:
    #Q-Learning 方法解决
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
    def greedy_policy(self, q, s):
        # 定义贪婪策略
        a_max = q[s, :].argmax()  # 取的是表格中最大q对应动作的列号
        return self.actions[a_max]
    def epsilon_greedy_policy(self, q, s, epsilon):
        # 定义epsilon-贪婪策略
        a_max = q[s, :].argmax()
        if np.random.uniform() < 1-epsilon:  # np包下random子库 uniform(low,high,size)产生low起始，high结束，size形状均匀分布
            return self.actions[a_max]
        else:
            return self.actions[int(random.random()*len(self.actions))]
        pass
    def plot_success_num(self,traj_num):#画轨迹数-累计成功轨迹数的图
        success_num=np.zeros(traj_num)#储存累计成功数 
        for i in range(self.success_traj[0]):
            success_num[i]=0
            pass
        count=1
        for traj in self.success_traj:
            success_num[traj]+=count
            count+=1
        for i in range((traj_num-1),(self.success_traj[0]-1),-1):
            if success_num[i]==0:
                success_num[i]=success_num[i+1]
                pass
            pass
        print(success_num)
        plt.figure(figsize=(80,60))
        x= range(traj_num)
        y= success_num[x]
        plt.plot(x,y)
        plt.show()
        pass
    def Q_method(self, traj_num,alpha,epsilon):
        #Q学习方法
        #Q_new(s,a) <- Q_old(s,a) + alpha *( R(s,a) + gamma * max_{a'}[Q_old(s',a')] - Q_old(s.a) )
        #a由b(s):epsilon贪婪策略获得
        #a'直接选取最大Q值对应动作
        self.Qvalue=np.zeros((len(self.states),len(self.actions)))
        self.success_traj=[]#记录成功的轨迹号
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
                a_next = self.greedy_policy(self.Qvalue,s_next)#与Sarsa同轨策略不同之处
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
                    self.success_traj.append(traj)
                    pass
                pass
            pass
        return self.Qvalue
    
if __name__=="__main__":
    traj_num=5000
    epsilon=0.1
    alpha=0.5
    grid=GridWindEnv()
    train_Q=Q_Gridworld(grid)
    train_Q.Q_method(traj_num,alpha,epsilon)
    print(train_Q.Qvalue)
    print(train_Q.success_traj)
    train_Q.plot_success_num(traj_num)
    grid.Qtable=train_Q.Qvalue
    flag=0
    path=[]
    step_num=0
    #s=30
    s=grid.rest()
    while flag==0 and step_num<30:
        grid.plane_position=grid.state_to_position(s)
        a=train_Q.greedy_policy(grid.Qtable,s)
        s_next,r,flag=grid.transform(s,a)
        step_num+=1
        path.append(s)
        grid.path=path
        grid.render()
        s=s_next
        pass
    grid.path.append(s)
    print(grid.path)
    print(len(train_Q.success_traj))
    while True:
        grid.render()
        pass
   
    pass
pass