import pygame
import random
import numpy as np
from loadpicwind import *
from pygame import *

class GridWindEnv:
    #带有风的格子世界，风在每步只作用一次
    def __init__(self):
        self.FPSLOCK=pygame.time.Clock()
        self.states=[]#70个格子，状态编号0-69
        for i in range(70):
            self.states.append(i)
            pass
        self.actions=['STOP','N','S','W','E','NW','NE','SW','SE']#停止、上、下、左、右、左上，右上，左下，右下
        self.path=[]#用于渲染轨迹
        self.Qtable=np.zeros((70,9))
        self.screen_size=(160*10,130*7)
        self.start_position=(0,3*130)
        self.limit_distance_x=150
        self.limit_distance_y=130
        self.target_position=(7*160,3*130)
        self.plane_position=(0,3*130)
        self.viewer=None
        self.gamma=0.9
        self.wind_state=[3,4,5,6,7,8,13,14
                        ,15,16,17,18,23
                        ,24,25,26,27,28
                        ,33,34,35,36,37
                        ,38,43,44,45,46
                        ,47,48,53,54,55
                        ,56,57,58,63,64,
                        65,66,67,68]
        #风强正数=向上
        #风强为1的
        self.wind_state_1=[3,4,5,8,13,14
                           ,15,18,23,24,25
                           ,28,33,34,35,38
                           ,43,44,45,48,53
                           ,54,55,58,63,64,65,68]
        #风强为2的
        self.wind_state_2=[6,7,16,17,26,27,36,
                           37,46,47,56,57,66,67]
        self.windpower=dict()
        #固定强度
        for i in self.wind_state_1:
            self.windpower[i]=1
            pass
        for j in self.wind_state_2:
            self.windpower[j]=2
            pass
        pass
    def out_of_range(self, position):
        if position[0]>160*9 or position[0]<0 or position[1]>6*130 or position[1]<0:
            flag=1
            pass
        else:
            flag=0
            pass
        return flag
    def find(self, position):
        flag=0
        if (abs(position[0]-self.target_position[0])<self.limit_distance_x ) and (abs(position[1]-self.target_position[1])<self.limit_distance_y):
            flag=1
            pass
        return flag
    def state_to_position(self, state):
        #状态中位置是格子号，从0-69，将格子号转化为格子左上角的坐标
        i=int(state/10)
        j=state%10
        position=[0,0]
        position[0]=160*j
        position[1]=130*i
        return position
    def state_to_grid(self, state):
        #将状态(0-69格子号）转换为第i行第j个格子
        i=int(state/10)
        j=state%10
        return j,i
    def position_to_state(self, position):
        #将格子左上角的坐标转换为格子号
        i=position[0]/160
        j=position[1]/130
        state=int(j*10+i)
        return state
    def rest(self):
        flag=1#不能放在终点也不能放在风中
        while flag==1:
            start_state=self.states[int(random.random()*len(self.states))]
            flag=self.find(self.state_to_position(start_state))
            if start_state in self.wind_state:
                flag=1
                pass
            pass
        return start_state
    def transform(self, state,action):
        #返回下一状态的标号、奖励、是否终止
        #智能体执行完动作后，考虑落地位置风的作用一次形成最终状态转移
        current_position=self.state_to_position(state)
        next_position=[0,0]
        flag_out_of_range=self.out_of_range(self.state_to_position(state))
        flag_find=self.find(self.state_to_position(state))
        if (flag_out_of_range==1) or (flag_find==1):
            return state,state,0,True
        if action=='STOP':
            next_position[0]=current_position[0]
            next_position[1]=current_position[1]        
        if action=='N':
            next_position[0]=current_position[0]
            next_position[1]=current_position[1]-130
            pass
        if action=='S':
            next_position[0]=current_position[0]
            next_position[1]=current_position[1]+130
            pass
        if action=='W':
            next_position[0]=current_position[0]-160
            next_position[1]=current_position[1]
            pass
        if action=='E':
            next_position[0]=current_position[0]+160
            next_position[1]=current_position[1]
            pass
        if action=='NW':
            next_position[0]=current_position[0]-160
            next_position[1]=current_position[1]-130
            pass
        if action=='NE':
            next_position[0]=current_position[0]+160
            next_position[1]=current_position[1]-130
            pass
        if action=='SW':
            next_position[0]=current_position[0]-160
            next_position[1]=current_position[1]+130
            pass
        if action=='SE':
            next_position[0]=current_position[0]+160
            next_position[1]=current_position[1]+130
            pass
        #最后风作用一次
        next_state=self.position_to_state(next_position)
        if next_state in self.wind_state:
            next_position[0]=next_position[0]
            next_position[1]=next_position[1]-130*self.windpower[next_state]
            pass
        flag_out_of_range=self.out_of_range(next_position)
        flag_find=(self.find(next_position)) 
        if flag_out_of_range==1:
            return state,-100,True
        if flag_find==1:
            return self.position_to_state(next_position),10,True
        return self.position_to_state(next_position),-1,False
    def gameover(self):
        for event in pygame.event.get():
            if event.type==QUIT:
                exit()
            pass
        pass
        
    def render(self):
        if self.viewer==None:
            pygame.init()
            self.viewer=pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Gridworld")
            self.plane=load_plane()
            self.wind=load_wind()
            self.target=load_target()
            self.background=load_background()
            self.viewer.blit(self.background,(0,0))
            self.font=pygame.font.SysFont('times',15)
            pass
        self.viewer.blit(self.background,(0,0))
        for i in range(11):
            pygame.draw.lines(self.viewer,(255,255,255),True,((i*160,0),(i*160,7*130)),2)#画竖线                              
            pass
        for i in range(8):
            pygame.draw.lines(self.viewer,(255,255,255),True,((0,i*130),(10*160,i*130)),2)#画横线
            pass
        self.viewer.blit(self.plane,self.plane_position) 
        if self.position_to_state(self.target_position) in self.wind_state:
            self.wind_state.remove(self.position_to_state(self.target_position))#去掉目标点的风图片
            pass
        for state in self.wind_state:
            self.viewer.blit(self.wind,self.state_to_position(state))
            pass
        self.viewer.blit(self.target,self.target_position)
        for state in range(70):
            j,i=self.state_to_grid(state)#第i行 第j个格子
            surface=self.font.render(str(round(float(self.Qtable[state,1]),2)),True,(0,0,0))#打印N的q
            self.viewer.blit(surface,((j*160+65),(i*130+10)))
            surface=self.font.render(str(round(float(self.Qtable[state,2]),2)),True,(0,0,0))#打印S的q
            self.viewer.blit(surface,((j*160+65),(i*130+110)))
            surface=self.font.render(str(round(float(self.Qtable[state,3]),2)),True,(0,0,0))#打印W的q
            self.viewer.blit(surface,((j*160+15),(i*130+60)))
            surface=self.font.render(str(round(float(self.Qtable[state,4]),2)),True,(0,0,0))#打印E的q
            self.viewer.blit(surface,((j*160+115),(i*130+60)))
            surface=self.font.render(str(round(float(self.Qtable[state,5]),2)),True,(0,0,0))#打印NW的q
            self.viewer.blit(surface,((j*160+15),(i*130+10)))
            surface=self.font.render(str(round(float(self.Qtable[state,6]),2)),True,(0,0,0))#打印NE的q
            self.viewer.blit(surface,((j*160+115),(i*130+10)))
            surface=self.font.render(str(round(float(self.Qtable[state,7]),2)),True,(0,0,0))#打印SW的q
            self.viewer.blit(surface,((j*160+15),(i*130+110)))
            surface=self.font.render(str(round(float(self.Qtable[state,8]),2)),True,(0,0,0))#打印SE的q
            self.viewer.blit(surface,((j*160+115),(i*130+110)))
            surface=self.font.render(str(round(float(self.Qtable[state,0]),2)),True,(0,0,0))#打印STOP的q
            self.viewer.blit(surface,((j*160+65),(i*130+60)))
            pass
        #画路径点
        for i in range(len(self.path)):
            rec_position=self.state_to_position(self.path[i])
            pygame.draw.rect(self.viewer,[255,0,0],[rec_position[0],rec_position[1],160,130],3) #路径显示是在格子上画矩形
            surface=self.font.render("Step"+str(i),True,(255,0,0))#路径方格上显示这是第几步
            self.viewer.blit(surface,(rec_position[0]+5,rec_position[1]+2))
            pass
        pygame.display.update()
        self.gameover()
        self.FPSLOCK.tick(30)
        pass
#TEST    
if __name__=="__main__":
    grid=GridWindEnv()
    s=36
    a='N'
    s_next,s_temp,r,t=grid.transform(s,a)
    print(s_temp)
    print(s_next)
    print(r)
    print(t)
    while True:
        grid.render()
        pass
