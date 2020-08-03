import random
import time
from yuanyang import *
class DP_Value_Iter:
    def __init__(self,yuanyang):
        self.states=yuanyang.states
        self.actions=yuanyang.actions
        self.v=[0.0 for i in range(len(self.states))] 
        self.pi=dict()
        self.gamma=yuanyang.gamma
        for state in self.states:
            flag1=0
            flag2=0
            flag1=yuanyang.collide(yuanyang.state_to_position(state))
            flag2=yuanyang.find(yuanyang.state_to_position(state))
            if flag1==1 or flag2==1:
                continue
            self.pi[state]=self.actions[int(random.random()* len(self.actions))]
            pass
        pass
    def value_iter(self):
        for i in range(500):
            delta=0.0
            for state in self.states:
                flag1=0
                flag2=0
                flag1=yuanyang.collide(yuanyang.state_to_position(state))
                flag2=yuanyang.find(yuanyang.state_to_position(state))
                if flag1==1 or flag2==1:continue
                a1=self.pi[state]
                next_s,r,t=yuanyang.transform(state,a1)
                #做一步策略评估
                v1=r+self.gamma*self.v[next_s]
                #立即策略改善
                for action in self.actions:
                    next_s,r,t=yuanyang.transform(state,action)
                    if v1<r+self.gamma*self.v[next_s]:
                        v1=r+self.gamma*self.v[next_s]
                        a1=action
                        pass
                    delta+=abs(v1-self.v[state])
                    #更新更有价值
                    self.v[state]=v1
                    self.pi[state]=a1
                    pass
            if delta<1e-6:
                print("****价值迭代进行：第 " +str(i)+" 次后完成****")
                break
            pass                          
        pass
    pass
#-----------测试脚本------------
if __name__=="__main__":
    yuanyang=YuanYangEnv()
    train=DP_Value_Iter(yuanyang)
    train.value_iter()
    s=yuanyang.reset()
    path=[]
    #向游戏中写入值函数
    for state in train.states:
        j,i=yuanyang.state_to_grid(state)
        yuanyang.value[j,i]=train.v[state]
        pass
    flag=1#渲染开关
    step_num=0
    while flag:
        path.append(s)
        yuanyang.path=path
        a=train.pi[s]
        print("在第"+str(s)+"格执行动作"+str(a))
        yuanyang.bird_male_position=yuanyang.state_to_position(s)
        yuanyang.render()
        #TODO为什么我加入time.sleep(?)方法就报错
        step_num+=1
        s_next,r,t=yuanyang.transform(s,a)
        if t==True or step_num>20:
            flag=0
            pass
        s=s_next
        pass
    yuanyang.bird_male_position=yuanyang.state_to_position(s)
    yuanyang.path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
    pass
pass