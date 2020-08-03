from MC_yuanyang import *
import matplotlib.pyplot as plt


class MC_RL:
    # 用于基于蒙特卡洛的方法解决yuanyang问题
    # 用于完成以下两个试验：
    # 1.试探性出发条件下首次访问型确定性的贪婪策略的“分幕式”策略迭代方法  （因为还是先预计该幕的Q表，然后再根据Q表选动作）
    # 2.同轨策略的MC控制方法，软策略，依然是一种“分幕式”策略迭代

    def __init__(self, yuanyang):
        self.Qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions)))*0.1  # 为什么要乘以0.1
        # 这是存放次数，为何初始次数不是0?因为避免除以0
        self.n = np.ones((len(yuanyang.states), len(yuanyang.actions)))*0.001
        self.actions = yuanyang.actions
        self.states = yuanyang.states
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang  # 为后面的方法省一个参数传递
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
            return self.action[int(random.random(len(self.actions)))]
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
            if self.mc_test() == 1:
                # self.mc_test用来判断目前的策略是否能找到雌鸟，利用Q表格选取动作然后仿真检验
                print("经过"+str(traj)+"条试探性出发的轨迹后，策略成功！")
                break
            # 下面进行试验
            while done == False and step_num < 30:
                # 开始生成一条轨迹，轨迹终止条件是用yuanyang的transform方法返回标志位碰撞或找到，另外步数过多也会强行停止
                # 首先与环境交互
                # S0->A0-> S1+R1 ->A1-> S2+R2 ... ->ST-1+RT-1  ->AT ->ST + RT 终止
                #ST是第一次撞墙或找到 这两种情况下的状态
                s_next, r, done = self.yuanyang.transform(s, a)
                a_number = self.action_to_num(a)
                # 为防止智能体往回走，加一个往回走更大惩罚
                if s_next in s_sample:
                    r = -2
                    pass
                # 存储经验
                s_sample.append(s) #S0~ST-1 T个状态 没有记录最终状态
                a_sample.append(a_number) #A0~AT-1 T个动作 同样也没有最终状态的动作
                r_sample.append(r) #R1~RT T个即时奖励 对应S1~ST状态下获得的
                #print(r_sample)#FIXME测试用
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
            # 计算终态ST-1的G值，由于达到终态后不转移，所以它后续无奖励
            #ST-1的G值实际就是Q(ST-1,AT-1) S,A分别对应于S,A最后一个采样
            g = self.Qvalue[s_sample[-1],a_sample[-1]]
            for i in range(len(s_sample)-1, -1, -1):
                # G(s)=r + gamma* G(s')
                g= (self.yuanyang.gamma) * g + r_sample[i]
                # 从倒数第二个时间开始
                # 用时间下标可以对应取到动作，而直接遍历元素则不行
                self.n[s_sample[i], a_sample[i]] += 1.0  # 统计各类二元组出现的次数，注意这个是在不同轨迹之间累计的
                # Q_new=Q_old + 1\N *(G_new - Q_old)
                self.Qvalue[s_sample[i], a_sample[i]] = self.Qvalue[s_sample[i], a_sample[i]] \
                    + (1 / (self.n[s_sample[i], a_sample[i]])) * (g-self.Qvalue[s_sample[i], a_sample[i]])
                pass
            pass
        return self.Qvalue
        
    def MC_RL_OnPolicy(self, num_traj, epsilon):
        #与MC_ES 分幕式策略迭代不同的是，在每幕选取动作时策略是一个概率型的，相同的是它们都是同轨策略
        # TODO 补全同轨策略MC控制方法
        
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