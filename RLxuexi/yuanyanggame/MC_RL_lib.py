from MC_yuanyang import *
import matplotlib.pyplot as plt


class MC_RL:
    # 用于基于蒙特卡洛的方法解决yuanyang问题
    # 用于完成以下两个试验：
    # 1.试探性出发条件下首次访问型确定性的贪婪策略的“分幕式”策略迭代方法  （因为还是先预计该幕的Q表，然后再根据Q表选动作）
    # 2.同轨策略的MC控制方法，软策略，依然是一种“分幕式”策略迭代
    """ ！！！原书中Q表更新是从前往后算，在Epsilon-贪婪策略中，因为试验次数有限，这样轨迹前面的值很好，
         后面的值不准。我这里是从后往前算，这样后面的值很好!!!
        在这个蒙特卡洛方法程序里。如果雄鸟出生点在格子偏右位置，就能正常工作，如果在靠前位置则不能"""
    """!!!为了加速训练，我把撞墙惩罚设置为-100，因为原先-10与走多次的-2相比太小了!!!"""
    def __init__(self, yuanyang):
        self.Qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions)))*0.1  # 为什么要乘以0.1
        # 这是存放次数，为何初始次数不是0?因为避免除以0
        self.n = np.ones((len(yuanyang.states), len(yuanyang.actions)))*0.001
        self.actions = yuanyang.actions
        self.states = yuanyang.states
        self.gamma = yuanyang.gamma
        self.yuanyang=yuanyang
        pass

    def greedy_policy(self, q, s):
        # 定义贪婪策略
        a_max = q[s, :].argmax()  # 取的是表格中最大q对应动作的列号
        return self.actions[a_max]

    def epsilon_greedy_policy(self, q, s, epsilon):
        # 定义epsilon-贪婪策略
        a_max = q[s, :].argmax()
        if np.random.uniform() < 1-epsilon:  # np包下random字库 uniform(low,high,size)产生low起始，high结束，size形状均匀分布
            return self.actions[a_max]
        else:
            return self.actions[int(random.random()*len(self.actions))]
        pass

    def action_to_num(self, a):
        # 返回动作的代号
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i
            pass
        pass

    def MC_learning_ES(self, num_traj):
        # 试探性出发MC（MC-ES)，需要指定从随机初始状态出发，蒙特卡洛试验生成多少条轨迹
        # 为确保重复使用，这里再初始化一次Q表和二元组次数记录表
        # 返回值是Q表
        # 若能够完成任务，则退出
        self.Qvalue = np.zeros(
            (len(self.yuanyang.states), len(self.yuanyang.actions)))*0.1
        self.n = np.ones((len(self.yuanyang.states),len(self.yuanyang.actions)))*0.001
        for traj in range(num_traj):
             # 每条轨迹作为存储经验，对应的都是第K步的当前状态，当前拟执行动作，当前收获的奖励
            s_sample = []
            a_sample = []
            r_sample = []
            # 随机选择一个起点，环境已经自带了这个功能
            s = self.yuanyang.reset()
            #print("起点为："+str(s))
            # 随机选择一个初始动作,这样就满足了试探性假设
            a = self.actions[int(random.random()*len(self.actions))]
            done = False
            step_num = 0  # 记录步数
            # if self.mc_test() == 1:
            #     # self.mc_test用来判断目前的策略是否能找到雌鸟，利用Q表格选取动作然后仿真检验
            #     print("经过"+str(traj)+"条试探性出发的轨迹后，测试从指定起点出发的策略成功！")
            #     break
            # 下面产生轨迹
            while done == False and step_num < 30:
                # 开始生成一条轨迹，轨迹终止条件是用yuanyang的transform方法返回标志位碰撞或找到，另外步数过多也会强行停止
                # 轨迹说明：
                # (S_0)-->A_0-->(S_1,R_1)-->A_1-->(S_2,R_2)... (临界S_T-1,R_T-1) -->A_T-1-->(R_T,      S_T撞)
                # |-----------------SAVE to SAMPLE------------------------------------------------|-- s=s_next
                s_next, r, done = self.yuanyang.transform(s, a)
                a_number = self.action_to_num(a)
                # 存储经验
                s_sample.append(s) #S0~ST-1 T个状态 没有记录最终状态
                a_sample.append(a_number) #A0~AT-1 T个动作 最后一个动作将状态驱动到S_T
                r_sample.append(r) #R1~RT T个延迟奖励 对应S0~ST-1状态下获得的
                step_num += 1
                s = s_next
                # 这里动作选取依据的是贪婪策略，尽量“同轨”，注意由于贪婪策略函数没有对存在多个相同最大值情况考虑，
                # 所以一开始肯定是卡死在一些位置，需要经过很多次试验。
                # 其实可以定一个自定的产生轨迹用的策略
                a = self.greedy_policy(self.Qvalue, s_next)
                #a=self.actions[int(random.random()*len(self.actions))]#完全随机的离轨采样
                pass
            # 试验结束后再价值迭代：Q_new=Q_old + 1\N *(G_new - Q_old)
            # G(s)=r + gamma* G(s')
            g = 0
            #g=self.Qvalue[s,self.action_to_num(a)]
            for i in range(len(s_sample)-1, -1, -1):
                # G(s)=r + gamma* G(s')
                g= (self.yuanyang.gamma) * g + r_sample[i]
                self.n[s_sample[i], a_sample[i]] += 1.0 
                # Q_new=Q_old + 1\N *(G_new - Q_old)
                self.Qvalue[s_sample[i], a_sample[i]] = self.Qvalue[s_sample[i], a_sample[i]] \
                    + (1 / (self.n[s_sample[i], a_sample[i]])) * (g-self.Qvalue[s_sample[i], a_sample[i]])
                pass
            pass
        return self.Qvalue
        
    def MC_RL_OnPolicy(self, num_traj, epsilon):
        #与MC_ES 分幕式策略迭代不同的是，轨迹都是从固定的初态开始在每幕选取动作时策略是一个概率型的
        # 相同的是它们都是同轨策略
        # 没有单独编制测试环节，测试融合在同轨策略实施过程中了
        #输入：用于学习的生成轨迹的次数、探索率epsilon
        #返回：Q表
        self.Qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))*0.1
        self.n = np.ones((len(self.yuanyang.states),len(self.yuanyang.actions)))*0.001
        
        for traj in range(num_traj):
            #和前面一样，取多条轨迹
            s_sample = []
            a_sample = []
            r_sample = []
            done=False
            step_num=0
            s=0 #固定初始状态
            #s=self.yuanyang.reset()#设定一个随机的初始状态
            #epsilon需要更新吗？
            #epsilon=epsilon*np.exp(-num_traj/1000)
                # 开始生成一条轨迹，轨迹终止条件是用yuanyang的transform方法返回标志位碰撞或找到，另外步数过多也会强行停止
                # 轨迹说明：
                # (S_0)-->A_0-->(S_1,R_1)-->A_1-->(S_2,R_2)... (临界S_T-1,R_T-1) -->A_T-1-->(R_T,      S_T撞)
                # |-----------------SAVE to SAMPLE------------------------------------------------|-- s=s_next
            while done==False and step_num<30:
                a=self.epsilon_greedy_policy(self.Qvalue,s,epsilon)
                s_next,r,done=self.yuanyang.transform(s,a)
                a_num=self.action_to_num(a)
                s_sample.append(s)
                a_sample.append(a_num)
                """ if s_next in s_sample:
                    r+=-2*s_sample.count(s_next)#这里想用一个“惩罚加深”来限制往回走
                    pass """
                r_sample.append(r)
                s=s_next
                """ if self.yuanyang.find(self.yuanyang.state_to_position(s))==1:
                    print("在第"+str(traj)+"次轨迹生成过程中达成了目标")
                    pass """
                step_num+=1
                pass
                #上面轨迹生成结束后，检查最后一个状态（其实未转移，只是记录）
            #该轨迹（幕）结束后才进行Q表估计和策略改善，和上面过程类似
            g=0
            
            for i in range(len(s_sample)-1, -1, -1):
                # G(s)=r + gamma* G(s')
                g= (self.yuanyang.gamma) * g + r_sample[i]
                # 从T-1时间开始
                # 用时间下标可以对应取到动作，而直接遍历元素则不行

                self.n[s_sample[i], a_sample[i]] += 1.0  # 统计各类二元组出现的次数，注意这个是在不同轨迹之间累计的
                # Q_new=Q_old + 1\N *(G_new - Q_old)
                self.Qvalue[s_sample[i], a_sample[i]] = self.Qvalue[s_sample[i], a_sample[i]] \
                    + (1 / (self.n[s_sample[i], a_sample[i]])) * (g-self.Qvalue[s_sample[i], a_sample[i]])
                pass
            pass
        return self.Qvalue
        pass
    def mc_test(self):
        # 用来测试使用当前的Q表格贪婪选取动作后是否能找到雌鸟
        # 返回flag表示是否找到
        # 注意！测试中起点人需要在下面人为指定，这里认为起点就是0状态
        s=0  # 测试起点
        flag=0  # 是否通过测试的标志位
        step=0  # 步数计数器
        done=False  # 是否继续步进的控制位，如果找到雌鸟或者撞墙就不再步进
        while done == False and step < 30:
            a=self.greedy_policy(self.Qvalue, s)  # 依据Q表格，贪婪策略选取下一个动作
            s_next, r, done=self.yuanyang.transform(s, a)
            s=s_next
            step += 1
            # 使用yuanyang类中find方法判断
            if self.yuanyang.find(self.yuanyang.state_to_position(s)) == 1:
                flag=1
                break
            pass
        return flag

#这是用来测试MC-ES的代码    
if __name__=="__main__":
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