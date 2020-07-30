
import random
import numpy as np
import pygame
from pygame import *
from load import *

class YuanYangEnv:
    #这是一个gridworld类游戏环境，雄鸟需要避开障碍物去找到雌鸟
    #游戏结束的条件是碰壁（撞墙或越界），或者找到雌鸟
    #环境包括实例化初始设置，格子世界生成，MDP模型(S,A,P,R)，以及折扣因子gamma
    #
    
    #------------------------实例化---------------------------------
    def __init__(self):
        self.states=[]
        for i in range(0,100):                  #状态0-99，注意range()方法中stop是达不到的
            self.states.append(i)
            pass
        self.path=[]
        self.actions=['up','down','left','right']
        self.gamma=0.8
        self.value=np.zeros((10,10))      #初始化v(s)
        self.viewer=None                        #viewer指渲染器，最初设置为空表示不立即渲染
        self.FPSCLOCK=pygame.time.Clock()       #TODO这啥意思？？
        self.screen_size=(1200,900)             #游戏窗口尺寸1200*900
        self.bird_position=(0,0)                #雄鸟的位置坐标
        self.limit_distance_x=120               #横向每步走120像素
        self.limit_distance_y=90                #纵向每步走120像素
        self.obstacle_size=(120,90)             #一个障碍物占满一个格子
        self.obstacle1_x=[]                     #游戏中共有两堵墙，每堵墙有8个障碍物组成，这里是每堵墙的每个障碍物左上顶点坐标集合列表
        self.obstacle1_y=[]
        self.obstacle2_x=[]
        self.obstacle2_y=[]
        for i in range(8):                      #创建由障碍物组成的两堵墙
            self.obstacle1_x.append(4*90)
            if i<=3:
                self.obstacle1_y.append(i*90)
            else:
                self.obstacle1_y.append((i+2)*90)
                pass
            self.obstacle2_x.append(8*90)
            if i<=4:
                self.obstacle2_y.append(i*90)
                pass
            else:
                self.obstacle2_y.append((i+2)*90)
                pass
            pass
        self.bird_male_init_position=[0,0]      #雄鸟初始位置，位置是图片的左上角，下同
        self.bird_male_position=[0,0]           #雄鸟当前位置
        self.bird_female_postion=[9*120,0]       #雌鸟位置
        pass
    
    #------------------------外环境机制---------------------------------
    def collide(self, state_position):
        #碰撞检测函数，输入一个坐标，返回是否碰撞
        flag=1                                  #是否碰撞总
        flag1=1                                 #是否与墙1碰撞
        flag2=1                                 #是否与墙2碰撞
        #判断与墙1碰撞
        dx=[]                                   #dx dy代表与所有障碍物横纵距离，下同
        dy=[]
        for i in range(8):                      #对于输入状态，检测与墙1中4个障碍物之间的距离
            dx1=abs(self.obstacle1_x[i]-state_position[0])
            dx.append(dx1)
            dy1=abs(self.obstacle1_y[i]-state_position[1])
            dy.append(dy1)
            pass
        mindx1=min(dx)
        mindy1=min(dy)
        if (mindx1>=self.limit_distance_x) or (mindy1>=self.limit_distance_y):
            flag1=0
            pass
        second_dx=[]
        second_dy=[]
        for i in range(8):                      #对于输入状态，检测与墙2中4个障碍物之间的距离
            dx2=abs(self.obstacle2_x[i]-state_position[0])
            second_dx.append(dx2)
            dy2=abs(self.obstacle2_y[i]-state_position[1])
            second_dy.append(dy2)
            pass
        mindx2=min(second_dx)
        mindy2=min(second_dy)
        if (mindx2>=self.limit_distance_x) or (mindy2>=self.limit_distance_y) :
            flag2=0
            pass
        if (flag1==0) and (flag2==0):           #与墙1碰撞和墙2碰撞都算碰撞
            flag=0
            pass
        if (state_position[0]>120*9) or (state_position[0]<0) or (state_position[1]>90*9) or (state_position[1]<0):
            flag=1                              #越界也算碰撞   
            pass
        return flag
        
    def find(self, state_position):
        #判断并返回雄鸟是否找到了雌鸟，位置坐标差小于最小运动步长即可（就是重叠了）
        flag=0
        if (abs(state_position[0]-self.bird_female_postion[0])<self.limit_distance_x) and (abs(state_position[1]-self.bird_female_postion[1])<self.limit_distance_y):
            flag=1    
            pass
        return flag
    
    def state_to_position(self, state):
        #状态中位置是格子号，从0-99，将格子号转化为格子左上角的坐标
        i=int(state/10)
        j=state%10
        position=[0,0]
        position[0]=120*j
        position[1]=90*i
        return position
    
    def state_to_grid(self, state):
        #将状态(0-99格子号）转换为第i行第j个格子
        i=int(state/10)
        j=state%10
        return j,i
        
    def position_to_state(self, position):
        #将格子左上角的坐标转换为格子号
        i=position[0]/120
        j=position[1]/90
        state=int(j*10+i)
        return state
    
    def reset(self):
        #环境重置函数，将雄鸟放置在环境中，不能放到雌鸟位置，也不能放到障碍物上
        flag1=1
        flag2=1
        while flag1 or flag2 ==1:
            state=self.states[int(random.random()*len(self.states))]#随机放置在一个位置
            flag1=self.collide(self.state_to_position(state))#不能碰撞
            flag2=self.find(self.state_to_position(state))#不能直接放到雌鸟头上
            pass
        return state
    
    #------------------------MDP过程---------------------------------    
    def transform(self,state,action):
        #描述状态转移作用
        #输入 状态和动作
        #返回 转移后下一状态，奖励，游戏是否结束
        current_position=self.state_to_position(state)#先把状态变成坐标
        next_position=[0,0]#后面要检查雄鸟状态转移后是否碰撞及找到雌鸟
        flag_collide=0
        flag_find=0
        #状态转移前先判断当前是否已经碰撞或找到（终止态）
        #终止态不转移，回报为0，立即结束游戏
        flag_collide=self.collide(current_position)
        flag_find=self.find(current_position)
        if (flag_collide==1)or(flag_find==1):
            return state,0,True
        #--------------------状态转移机制----------------------
        #********************确定性版本************************************
        if action=='up':
            next_position[0]=current_position[0]
            next_position[1]=current_position[1]-90
            pass
        if action=='down':
            next_position[0]=current_position[0]
            next_position[1]=current_position[1]+90
            pass    
        if action=='left':
            next_position[0]=current_position[0]-120
            next_position[1]=current_position[1]
            pass
        if action=='right':
            next_position[0]=current_position[0]+120
            next_position[1]=current_position[1]
            pass
        #********************************************************
        
        
        #---------------------------------------------------
        #判断转移后是否碰撞或找到
        flag_collide=self.collide(next_position)
        flag_find=self.find(next_position)
        #下一状态为碰撞 奖励-1，下一状态为找到，奖励为1
        if (flag_collide==1):
            return self.position_to_state(current_position),-1,True
        if (flag_find==1):
            return self.position_to_state(next_position),1,True
        #下一状态为普通状态，奖励为0
        return self.position_to_state(next_position),0,False    
        
    def gameover(self):
        #TODOpygame事件获取判断是否结束游戏  不知道啥意思
        for event in pygame.event.get():
            if event.type==QUIT:
                exit()
    def render(self):
        #渲染游戏，如果没有窗口就画一个窗口
        if self.viewer is None:
            pygame.init()#不知道啥意思
            self.viewer=pygame.display.set_mode(self.screen_size,0,32)
            pygame.display.set_caption("yuanyang")
            #下载图片，图片加载器定义在load.py中
            self.bird_male=load_bird_male()
            self.bird_female=load_bird_female()
            self.background=load_background()
            self.obstacle=load_obstacle()
            #用pygame的blit函数把图片画在窗口里
            self.viewer.blit(self.bird_female,self.bird_female_postion)
            self.viewer.blit(self.background,(0,0))
            self.font=pygame.font.SysFont('times',15)
            pass
        #如果环境有了窗口，就开始画格子，把鸟放上去，以及把每个格子的值函数打上去
        self.viewer.blit(self.background,(0,0))
        
        #绘制游戏中图形：1.格子；2.鸟；3.墙；4.值函数打印；5.雄鸟行动路径
        
        #画格子
        for i in range(11):
            #先画竖线
            pygame.draw.lines(self.viewer,(255,255,255),True,((120*i,0),(120*i,900)),1)
            #再画横线
            pygame.draw.lines(self.viewer,(255,255,255),True,((0,90*i),(1200,90*i)),1)
        #画雌鸟
        self.viewer.blit(self.bird_female,self.bird_female_postion)
        #画雄鸟
        self.viewer.blit(self.bird_male,self.bird_male_position)
        #画2面墙
        for i in range(8):
            self.viewer.blit(self.obstacle,(self.obstacle1_x[i],self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle,(self.obstacle2_x[i],self.obstacle2_y[i]))
            pass
        #把每个格子的值函数打印上去
        for i in range(10):
            for j in range(10):
                #渲染数字
                surface=self.font.render("V="+str(round(float(self.value[i,j]),3)),True,(0,0,0))
                self.viewer.blit(surface,(120*i+5,90*j+70))
            pass
        #画路径点
        for i in range(len(self.path)):
            rec_position=self.state_to_position(self.path[i])#yuanyang.path是存放最终路径的列表
            pygame.draw.rect(self.viewer,[255,0,0],[rec_position[0],rec_position[1],120,90],3) #路径显示是在格子上画矩形
            surface=self.font.render("Step"+str(i),True,(255,0,0))#路径方格上显示这是第几步
            self.viewer.blit(surface,(rec_position[0]+5,rec_position[1]+5))
            pass
        pygame.display.update()
        self.gameover()
        # time.sleep(0.1)
        self.FPSCLOCK.tick(30)
        pass
    pass

#-------------------测试游戏环境显示效果----------------
if __name__=="__main__":
    yy=YuanYangEnv()
    yy.render()
    while True:
        for event in pygame.event.get():
            if event.type==QUIT:
                exit()
                pass
            pass
        pass
    pass
pass
#----------------------------------------------------