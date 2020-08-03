import random
import time
from yuanyang import YuanYangEnv

class DP_Policy_Iter:
    def __init__(self,yuanyang):
        #初始化时需要传入yuanyang游戏环境
        #初始化值函数v
        #用字典储存策略
        #yuanyang=YuanYangEnv()
        self.states=yuanyang.states
        self.actions=yuanyang.actions
        self.v=[0.0 for i in range(len(self.states))]#列表解析，当i访问到range中对象时，生成一个0.0
        self.pi=dict()
        self.gamma=yuanyang.gamma
        #初始化策略,当状态不是终止态时，随机化初始动作
        for state in self.states:
            flag1=0
            flag2=0
            flag1=yuanyang.collide(yuanyang.state_to_position(state))
            flag2=yuanyang.find(yuanyang.state_to_position(state))
            #如果是终止态，就跳过去初始化下一个状态
            if flag1==1 or flag2==1:
                continue
                pass
            #如果是普通状态，随机选择一个动作作为策略
            self.pi[state]=self.actions[int(random.random()*len(self.actions))]
            pass
       
        pass
    
    def Policy_Eva(self):
        #策略评估
        #与游戏世界交互中沿着决策树更新值函数
        #在这个格子世界游戏中，策略会指导在每个状态只会做出一个动作，所以策略评估公式中去掉求期望形式，直接通过次数迭代即可
        #yuanyang=YuanYangEnv()#
        for i in range(100):
            delta=0.0#一轮迭代结束后总累计误差
            for state in self.states:
                flag1=0
                flag2=0
                flag1=yuanyang.collide(yuanyang.state_to_position(state))
                flag2=yuanyang.find(yuanyang.state_to_position(state))
                if flag1==1 or flag2==1:
                    continue
                action=self.pi[state]#依据当前策略，选择当前状态下的动作
                next_s,r,t=yuanyang.transform(state,action)#返回下一状态，奖励，是否终止
                new_v=r+(self.gamma)*(self.v[next_s])
                delta+=abs(new_v-self.v[state])#误差累加器
                self.v[state]=new_v#更新值函数
                pass
            if delta<1e-6:
                print("****策略评估已迭代："+str(i)+" 次****")
                break
            pass
        pass
    
    def Policy_Improve(self):
        #策略改进
        #与游戏世界交互在状态下尝试各种动作，以最大化动作后值函数为目的改进原策略pi
        #yuanyang=YuanYangEnv()#
        #对状态进行遍历，终止态不处理
        #确定性策略，直接找使下个状态估计价值最大的动作
        for state in self.states:
            flag1=0
            flag2=0
            flag1=yuanyang.collide(yuanyang.state_to_position(state))
            flag2=yuanyang.find(yuanyang.state_to_position(state))
            if flag1==1 or flag2==1:
                continue
            #按顺序遍历尝试动作,然后两两比较
            #先考察一个动作的情况，暂存为最优值函数v1和最优动作a1
            a1=self.actions[0]
            next_s,r,t=yuanyang.transform(state,a1)
            v1=r+self.gamma*self.v[next_s]
            #然后考察后面动作的情况
            for a in self.actions:
                next_s,r,t=yuanyang.transform(state,a)
                #先不赋值，直接计算比较,如果转移后v大，那么就转移后的值函数和动作是最优
                if v1<=(r+self.gamma*self.v[next_s]):
                    a1=a
                    v1=r+self.gamma*self.v[next_s]
                    self.pi[state]=a1
                    pass
                pass
            pass
        pass
    
    def Policy_iter_main(self):
        #策略迭代算法：先策略评再估策略提升，总共迭代几次
        for i in range(100):
            self.Policy_Eva()
            pi_old=self.pi.copy
            self.Policy_Improve()
            #直到策略不变为止
            if pi_old==self.pi:
                print("策略改进完成，共用： "+str(i)+" 轮")
                break
            pass
        pass
    pass
#------------测试脚本---------------
if __name__=="__main__":
    #在yuanyang环境下测试训练效果，画出路径，打印值函数
    #起点随机
    flag=1#渲染路径点开关
    path=[]#路径临时变量
    step_num=0
    yuanyang=YuanYangEnv()
    s=yuanyang.reset()#设置初始状
    train=DP_Policy_Iter(yuanyang)
    train.Policy_iter_main()
    for state in range(100):
        j,i=yuanyang.state_to_grid(state)
        yuanyang.value[j,i]=train.v[state]
        pass
    while flag:
        #渲染路径点，相当于把雄鸟从初始状态按策略pi带入环境中再跑一遍
        path.append(s)
        yuanyang.path=path #在render中已定义了path，只需把路径传递给yuanyang类中的环境渲染器
        a=train.pi[s]
        print("在第"+str(s)+"格执行动作"+str(a))
        yuanyang.bird_male_position=yuanyang.state_to_position(s)
        #画一帧
        yuanyang.render()
        time.sleep(0.2)
        step_num+=1
        s_next,r,t=yuanyang.transform(s,a)
        #达到终止态或超时，把状态转移但是不渲染，也不传递路径，退出循环
        if t==True or step_num>=200:
            flag=0
            pass
        s=s_next
        pass
    #补上转移但不渲染的渲染步骤
    yuanyang.bird_male_position=yuanyang.state_to_position(s)
    #补上路径
    yuanyang.path.append(s)
    #渲染
    yuanyang.render()
    while True:
        yuanyang.render()
        pass
    pass
pass