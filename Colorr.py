#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tkinter as tk
from tkinter import Menu, Scale, Frame, Button, IntVar, Label, OptionMenu, Checkbutton
import tkinter.colorchooser as colorchooser
import numpy as np
import cv2
from PIL import Image, ImageTk
import csv
import os
from tkinter import messagebox

        

# 定义环境类
class Environment:
    def __init__(self, bg_color, brightness, texture=None, texture_intensity=0.1):
        self.bg_color = bg_color
        self.brightness = brightness
        self.texture = texture
        self.texture_intensity = texture_intensity
        self.contrast = 1.0
        self.saturation = 1.0

    # 渲染环境
    def render(self):
        image = np.full((500, 500, 3), self.bg_color, dtype=np.uint8)    # Create an image with a filled background colour
        if self.texture == 'noise':       # Add random noise textures  
            noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
            image = cv2.addWeighted(image, 1 - self.texture_intensity, noise, self.texture_intensity, 0)
        elif self.texture == 'stripes':     # Add striped texture    
            for i in range(0, image.shape[1], 10):
                cv2.line(image, (i, 0), (i, image.shape[0]), (255, 255, 255), 1)
        elif self.texture == 'dots':       # Add dots texture  
            for i in range(0, image.shape[1], 10):
                for j in range(0, image.shape[0], 10):
                    cv2.circle(image, (i, j), 1, (255, 255, 255), -1)
                    
                    
        # 调整图像亮度
        image = image * (self.brightness / 255.0)
        image = np.clip(image, 0, 255).astype(np.uint8)

        # 调整对比度和饱和度
        image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=0)  # 对比度
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV colour space for saturation adjustment
        s_channel = image[:, :, 1]
        s_channel = np.clip(s_channel * self.saturation, 0, 255)  # saturation
        image[:, :, 1] = s_channel
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # Convert back to BGR colour space
        return image
    

# 定义颜色块类
class ColorBlock:
    def __init__(self, color, environment, position, brightness=150, size=40):
        self.color = color
        self.original_color = color[:]  # 记录probe原始颜色
        self.environment = environment
        self.position = position
        self.brightness = brightness / 255  # 添加亮度属性，转换为比例
#         self.contrast = contrast if '2' in self.__class__.__name__ else 1.0  # 仅在ColorBlock2类中添加对比度属性
        self.size = size

    # 渲染颜色块
    def render(self):
        image = self.environment.render()    # 使用环境渲染方法渲染背景
#         color_with_brightness = tuple(int(i * self.brightness) for i in self.original_color)  # 考虑亮度调整颜色块的颜色
        color_with_brightness = tuple(int(i * self.brightness) for i in reversed(self.original_color))

        start_point = (self.position[0] - self.size//2, self.position[1] - self.size//2)    # 定义颜色块的起点，考虑size
        end_point = (self.position[0] + self.size//2, self.position[1] + self.size//2)      # 定义颜色块的终点，考虑size

        image = cv2.rectangle(image, start_point, end_point, color_with_brightness, -1)    # 在图像上绘制颜色块
#         if '2' in self.__class__.__name__:       # 在ColorBlock2类中使用对比度
#             image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=0)  
        return image

    
    # 设置颜色块的颜色
    def set_color(self, color):
        self.color = color
        self.original_color = color[:]  # 当probe颜色发生改变时，同时更新原始颜色
        


class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # 添加一个按钮来切换到第一页
        tk.Button(self, text="Go to page one", command=lambda: master.switch_frame(PageOne)).pack()          
        
       # 创建环境和色块
        self.env1 = Environment([200, 173, 196], 150)
        self.env2 = Environment([200, 173, 196], 150)
        self.probe1 = ColorBlock([116, 120, 122], self.env1, position=(250, 250), brightness=150)
        self.probe2 = ColorBlock([116, 120, 122], self.env2, position=(250, 250), brightness=150)
        
        

        # 用网格布局创建标签和滑动条
        self.frame = tk.Frame(self)
        self.label = tk.Label(self.frame)
        self.label.grid(row=0, column=0, columnspan=6)
        
        #用户帮助按钮
        tk.Button(self.frame, text=' ❓ ', command=self.show_help, height=10, width=1).grid(row=0, column=7)

        # 创建并添加按钮
        tk.Button(self.frame, text='Color of Probes', command=self.update_probes_color).grid(row=1, column=0)
        tk.Button(self.frame, text='Color of Environment 1', command=self.update_env1_color).grid(row=1, column=1)
        tk.Button(self.frame, text='Color of Environment 2', command=self.update_env2_color).grid(row=1, column=2)

        # 创建并添加env1纹理选项菜单
        self.env1_texture_var = tk.StringVar(self)
        self.env1_texture_var.set('none')
        self.texture_options = {'none', 'noise', 'stripes', 'dots'}
        tk.OptionMenu(self.frame, self.env1_texture_var, *self.texture_options, command=self.update_env1_texture).grid(row=1, column=3)
        
        # 创建并添加env2纹理选项菜单
        self.env2_texture_var = tk.StringVar(self)
        self.env2_texture_var.set('none')
        self.texture_options = {'none', 'noise', 'stripes', 'dots'}
        tk.OptionMenu(self.frame, self.env2_texture_var, *self.texture_options, command=self.update_env2_texture).grid(row=1, column=4)       

        
        # 创建并添加重置按钮
        tk.Button(self.frame, text='Reset', command=self.reset).grid(row=6, column=5)

        # 创建并添加滑动条
        tk.Label(self.frame, text="Environment 1 Brightness").grid(row=2, column=0)
        self.scale_env1_brightness = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, command=self.update_env1)
        self.scale_env1_brightness.grid(row=2, column=1)

        tk.Label(self.frame, text="Environment 2 Brightness").grid(row=2, column=2)
        self.scale_env2_brightness = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, command=self.update_env2)
        self.scale_env2_brightness.grid(row=2, column=3)

        tk.Label(self.frame, text="Probe 1 Brightness").grid(row=3, column=0)
        self.scale_probe1_brightness = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, command=self.update_probe1_brightness)
        self.scale_probe1_brightness.grid(row=3, column=1)

        tk.Label(self.frame, text="Probe 2 Brightness").grid(row=3, column=2)
        self.scale_probe2_brightness = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, command=self.update_probe2_brightness)
        self.scale_probe2_brightness.grid(row=3, column=3)
        
        
        tk.Label(self.frame, text="Environment 2 Contrast").grid(row=5, column=0)
        self.scale_env2_contrast = tk.Scale(self.frame, from_=0, to=2, resolution=0.1,
                                            orient=tk.HORIZONTAL, length=200,
                                            command=self.update_env2_contrast)
        self.scale_env2_contrast.grid(row=5, column=1)

        tk.Label(self.frame, text="Environment 2 Saturation").grid(row=5, column=2)
        self.scale_env2_saturation = tk.Scale(self.frame, from_=0, to=2, resolution=0.1, orient=tk.HORIZONTAL, length=200, command=self.update_env2_saturation)
        self.scale_env2_saturation.grid(row=5, column=3)

        
        # 创建并添加滑动条，调节rgb通道值，用于调节probe2的色温
        tk.Label(self.frame, text="Probe 2 Red").grid(row=4, column=0)
        self.scale_probe2_red = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         length=200, command=self.update_probe2_red)
        self.scale_probe2_red.grid(row=4, column=1)

        tk.Label(self.frame, text="Probe 2 Green").grid(row=4, column=2)
        self.scale_probe2_green = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                           length=200, command=self.update_probe2_green)
        self.scale_probe2_green.grid(row=4, column=3)

        tk.Label(self.frame, text="Probe 2 Blue").grid(row=4, column=4)
        self.scale_probe2_blue = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                          length=200, command=self.update_probe2_blue)
        self.scale_probe2_blue.grid(row=4, column=5)
        
        # 添加按钮，供用户选择
        tk.Button(self.frame, text='Probe1 Brighter', command=self.record_probe1_brighter, height=2, width=20,fg='#ad6598').grid(row=6, column=1)
        tk.Button(self.frame, text='Probe2 Brighter', command=self.record_probe2_brighter, height=2, width=20, fg='#ad6598').grid(row=6, column=2)
        tk.Button(self.frame, text='Similar', command=self.record_probe_similar, height=2, width=20, fg='#ad6598').grid(row=6, column=3)    

        self.set_default_values()
 
        self.frame.pack()

        self.update_image()

        
    def set_default_values(self):
        self.scale_env1_brightness.set(150) 
        self.scale_env2_brightness.set(150)
        self.scale_probe1_brightness.set(150) 
        self.scale_probe2_brightness.set(150)
        self.scale_probe2_red.set(116)
        self.scale_probe2_green.set(120)
        self.scale_probe2_blue.set(122)
        self.scale_env2_contrast.set(1.0)
        self.scale_env2_saturation.set(1.0)

    
        
    def show_help(self):
        help_window = tk.Toplevel(self)
        help_window.title("User Help")          # 设置窗口标题
        
        # 添加一些关于系统使用方法、实验目的和方法的文字
        usage_label = tk.Label(help_window, text="❓Purpose: To investigate the effects of various environmental factorson the perception of colour brightness.\n")
        usage_label.pack(anchor='w')

        StartPage_label = tk.Label(help_window, text="❓StartPage: Adjust the parameters freely and observe the brightness of the two probes.")
        StartPage_label.pack(anchor='w')

        PageOne_label = tk.Label(help_window, text="❓PageOne: Select different scenes via the drop-down menu and observe the brightness of the probes.\n")
        PageOne_label.pack(anchor='w')
        
        Method_label = tk.Label(help_window, text="❓Experimental method: Select the button corresponding to the probe that is considered brighter. (left: probe1, right: probe2)")
        Method_label.pack(anchor='w')
        
        Interface_label = tk.Label(help_window, text="❓Click the navigation button at the top of the page to switch the interface.")
        Interface_label.pack(anchor='w')
    
    
    def update_env1(self, _=None):      # Update the brightness of env1. Get value from scale_env1_brightness.
        self.env1.brightness = self.scale_env1_brightness.get()
        self.update_image()

    def update_env2(self, _=None):
        self.env2.brightness = self.scale_env2_brightness.get()
        self.update_image()

    def update_probes_color(self):    # 使用颜色选择器来更新探针的颜色
        color = colorchooser.askcolor(title ="Color of Probes")
        if color[1] is not None:     # 如果选择了有效的颜色，则转换为RGB值，并反转顺序。
#             new_color = [int(color[1][5:7], 16), int(color[1][3:5], 16), int(color[1][1:3], 16)]
            new_color = [int(color[1][1:3], 16), int(color[1][3:5], 16), int(color[1][5:7], 16)]

            self.probe1.set_color(new_color)  # 使用新的set_color方法更新probe1颜色
            self.probe2.set_color(new_color)
            
            # 更新2的RGB通道滑块值
            self.scale_probe2_red.set(new_color[0])
            self.scale_probe2_green.set(new_color[1])
            self.scale_probe2_blue.set(new_color[2])
            
            self.update_image()

    def update_probe1_brightness(self, value): 
            self.probe1.brightness = int(value) / 255
            self.update_image()

    def update_probe2_brightness(self, value):
            self.probe2.brightness = int(value) / 255
            self.update_image()

            
    
    def update_env1_color(self):
        color = colorchooser.askcolor(title ="Color of Environment 1")
        if color[1] is not None:
            new_color = [int(color[1][5:7], 16), int(color[1][3:5], 16), int(color[1][1:3], 16)]  # 反转颜色
            self.env1.bg_color = new_color
            self.update_image()

    def update_env2_color(self):
        color = colorchooser.askcolor(title ="Color of Environment 2")
        if color[1] is not None:
            new_color = [int(color[1][5:7], 16), int(color[1][3:5], 16), int(color[1][1:3], 16)]
            self.env2.bg_color = new_color
            self.update_image()
    
    def update_env1_texture(self, _=None):
        texture = self.env1_texture_var.get()
        if texture == 'none':
            self.env1.texture = None
        else:
            self.env1.texture = texture
        self.update_image()
    
    def update_env2_texture(self, _=None):
        texture = self.env2_texture_var.get()
        if texture == 'none':
            self.env2.texture = None
        else:
            self.env2.texture = texture
        self.update_image()

    
    def update_probe2_red(self, _=None):
        self.probe2.set_color([self.scale_probe2_red.get(),
                               self.probe2.color[1], self.probe2.color[2]])
        self.update_image()

    def update_probe2_green(self, _=None):
        self.probe2.set_color([self.probe2.color[0], self.scale_probe2_green.get(), self.probe2.color[2]])
        self.update_image()

    def update_probe2_blue(self, _=None):
        self.probe2.set_color([self.probe2.color[0], self.probe2.color[1], self.scale_probe2_blue.get()])
        self.update_image()

        
    def update_env2_contrast(self, value):
        self.env2.contrast = float(value)
        self.update_image()

    def update_env2_saturation(self, value):
        self.env2.saturation = float(value)
        self.update_image()
    
    
    def record_probe1_brighter(self):
        self.record_choice('Probe 1')
    def record_probe2_brighter(self):
        self.record_choice('Probe 2')
    def record_probe_similar(self):
        self.record_choice('Similar')
    
    def record_choice(self, brighter_probe):          # 记录当前所有的参数
        filename = 'user_choices.csv'
        if os.path.isfile(filename):           # 获取文件的行数（如果文件存在）
            with open(filename, 'r') as csvfile:
                row_count = sum(1 for row in csv.reader(csvfile)) - 1
        else:
            row_count = 0
            
        data = {
            'Index': row_count + 1,
            'Probe 1 Color': self.probe1.color,
            'Probe 2 Color': self.probe2.color,
            'Probe 1 Brightness': self.probe1.brightness,
            'Probe 2 Brightness': self.probe2.brightness,            
            'Environment 1 Color': self.env1.bg_color,
            'Environment 2 Color': self.env2.bg_color,
            'Environment 1 Brightness': self.env1.brightness,
            'Environment 2 Brightness': self.env2.brightness,
            'Env1_Texture': self.env1_texture_var.get(),
            'Env2_Texture': self.env2_texture_var.get(),
            'Environment 2 Contrast': self.env2.contrast,
            'Environment 2 Saturation': self.env2.saturation,

            'Brighter Probe': brighter_probe
        }
        
        # 将数据写入到文件中
        with open('user_choices.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = data.keys())
            if row_count == 0:   # 如果是第一行，则写入头部
                writer.writeheader()
            writer.writerow(data)
        
        
    def update_image(self):
        # Render the images of probe1 and probe2
        image1 = self.probe1.render()
        image2 = self.probe2.render()
        image = np.concatenate((image1, image2), axis=1)  # Splice the two images horizontally
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert the image from BGR to RGB
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)    # Convert the image to a format usable by tkinter
        self.label.config(image=image)     # Update the image labels
        self.label.image = image

    def reset(self):      # 重置参数
        self.scale_env1_brightness.set(150)
        self.scale_env2_brightness.set(150)
        
        self.scale_probe1_brightness.set(150)
        self.scale_probe2_brightness.set(150)
        
        self.env1.bg_color = [200, 173, 196]
        self.env2.bg_color = [200, 173, 196]
        
        reset_color = [116, 120, 122]
        self.probe1.set_color(reset_color) # 使用set_color方法更新probe1颜色
        self.probe2.set_color(reset_color) # 使用set_color方法更新probe2颜色

                
        self.scale_probe2_red.set(116)
        self.scale_probe2_green.set(120)
        self.scale_probe2_blue.set(122)
        
        self.scale_env2_saturation.set(1.0)
        self.scale_env2_contrast.set(1.0)

        self.update_image()

    def run(self):
        self.window.mainloop()


        
        
class PageOne(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # 回到起始页面的按钮
        tk.Button(self, text="Go back to start page",
                  command=lambda: master.switch_frame(StartPage)).grid(row=0, column=0, columnspan=3)

        # 创建用于显示probe的图像的标签
        self.label = tk.Label(self)
        self.label.grid(row=1, column=0, columnspan=3)        
        
        # 创建环境和色块
        self.env1 = Environment([190, 180, 230], 150)
        self.env2 = Environment([190, 180, 230], 150)
        self.probe1 = ColorBlock([200, 150, 200], self.env1, position=(250, 250), brightness=150)
        self.probe2 = ColorBlock([200, 150, 200], self.env2, position=(250, 250), brightness=150)

        tk.Label(self, text="Select a preset scene", font=('Helvetica', 15)).grid(row=2, column=0, columnspan=3)

        # 创建下拉菜单以选择预设场景
        self.scene_var = tk.StringVar()
        self.scene_var.set('Select a scene')
        self.scenes = [f'Scene {i}' for i in range(1, 17)]
        self.scene_menu = tk.OptionMenu(self, self.scene_var, *self.scenes, command=self.apply_scene)
        self.scene_menu.grid(row=3, column=0, columnspan=3)

        # 文本标签
        tk.Label(self, text="Please choose the brighter one:").grid(row=4, column=0, columnspan=3)
        
        # 新增三个按钮
        tk.Button(self, text="Probe 1", command=self.choose_probe1, padx=10, pady=10, fg='#ad6598').grid(row=5, column=1)
        tk.Button(self, text="Probe 2", command=self.choose_probe2, padx=10, pady=10, fg='#ad6598').grid(row=6, column=1)
        tk.Button(self, text="Similar", command=self.choose_similar, padx=10, pady=10, fg='#ad6598').grid(row=7, column=1)

        # 新增Finish按钮
        tk.Button(self, text="Finish", command=self.finish).grid(row=7, column=2, columnspan=3)

        
        # 初始化图像
        self.update_image()

        
    def apply_scene(self, scene_name):          # 根据选择的场景应用不同的环境参数
        
        # 恢复到初始设置
        self.env1.bg_color = [190, 180, 230]
        self.env2.bg_color = [190, 180, 230]
        self.env1.brightness = 150
        self.env2.brightness = 150
        self.env1.texture = None
        self.env2.texture = None
        self.env1.texture_intensity = 0.1
        self.env2.texture_intensity = 0.1
        self.probe1.color = [200, 150, 200]
        self.probe2.color = [200, 150, 200]
        
        
        scene_index = int(scene_name.split(' ')[1]) - 1
        
        # 背景亮度不同   
        if scene_index == 0:        # Scene1:The background brightness of the two probes is 220 and 110 respectively
            self.env1.brightness = 220
            self.env2.brightness = 110 
        elif scene_index == 1:        # Scene2: The background brightness is 180 and 165
            self.env1.brightness = 180
            self.env2.brightness = 165
        elif scene_index == 2:        # Scene3: The background brightness is 200 and 160
            self.env1.brightness = 200
            self.env2.brightness = 150  

        # 背景颜色不同 
        elif scene_index == 3:        # 两个probe的背景颜色分别为浅粉色和深蓝色
            self.env1.bg_color = [255, 182, 193]  # 浅粉色
            self.env2.bg_color = [0, 0, 128]      # 深蓝色          
        elif scene_index == 4:        # 两个probe的背景颜色分别为紫色和黄色
            self.env1.bg_color = [128, 0, 128] 
            self.env2.bg_color = [255, 255, 0]  
            
        # 背景纹理不同      
        elif scene_index == 5:        # probe2的背景添加噪点纹理
            self.env2.texture = 'noise'
            self.env2.texture_intensity = 0.1  # 纹理强度
        elif scene_index == 6:        # probe2的背景添加条纹纹理
            self.env2.texture = 'stripes'
            self.env2.texture_intensity = 0.1  # 纹理强度         
        
        # 光照强度不同
        elif scene_index == 7:        # 两个probe亮度分别为180和140
            self.probe1.brightness = 180/255
            self.probe2.brightness = 140/255
        elif scene_index == 8:        # 两个probe亮度分别为170和160
            self.probe1.brightness = 170/255
            self.probe2.brightness = 160/255        
        elif scene_index == 9:        # 两个probe亮度分别为170和145
            self.probe1.brightness = 170/255
            self.probe2.brightness = 145/255
        
        # 光照色温不同    
        elif scene_index == 10:        # 改变probe2的色温，使其更暖
            self.probe2.color = [230, 180, 170]  # 通过增加红色、绿色通道，稍微减少蓝色通道来实现  
        elif scene_index == 11:        # 改变probe2的色温，使其更冷
            self.probe2.color = [160, 110, 240]  # 通过增加蓝色通道，减少红色、绿色通道来实现
            
        # probe1、2背景饱和度不同
        elif scene_index == 12:     
            self.env2.saturation = 1.8      # probe2的背景饱和度为高饱和度，probe1背景饱和度不变，仍为1
        elif scene_index == 13:     
            self.env2.saturation = 0.3      # probe2的背景饱和度为低饱和度
            
        # probe1、2背景纹理都设置为条纹，但probe2背景对比度不同
        elif scene_index == 14:
            self.env1.texture = 'stripes'
            self.env1.texture_intensity = 0.1  # 纹理强度
            self.env2.texture = 'stripes'
            self.env2.texture_intensity = 0.1  # 纹理强度
            self.env2.contrast = 1.8      # probe2背景对比度为高对比度
        elif scene_index == 15:
            self.env1.texture = 'stripes'
            self.env1.texture_intensity = 0.1  # 纹理强度
            self.env2.texture = 'stripes'
            self.env2.texture_intensity = 0.1  # 纹理强度
            self.env2.contrast = 0.6      # probe2背景对比度为低对比度       
        

        # 更新图像
        self.update_image()

    def update_image(self):
        # 将probe的图像渲染到标签上
        image1 = self.probe1.render()
        image2 = self.probe2.render()
        image = np.concatenate((image1, image2), axis = 1)  # 横向拼接两个图像
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.label.config(image = image)
        self.label.image = image


    def choose_probe1(self):
        self.record_choice("Probe 1")
    def choose_probe2(self):
        self.record_choice("Probe 2")
    def choose_similar(self):
        self.record_choice("Similar")

    def record_choice(self, choice):
        scene_name = self.scene_var.get()
        
        # 将参数和选择存储为类属性
        self.choice_data = [
            scene_name,
            self.env1.bg_color,
            self.env2.bg_color,
            self.env1.brightness,
            self.env2.brightness,
            self.env1.texture if self.env1.texture else "None",
            self.env2.texture if self.env2.texture else "None",
            self.probe1.color,
            self.probe2.color,
            self.probe1.brightness,
            self.probe2.brightness,
            getattr(self.env2, 'saturation', 1), # 如果env2没有saturation属性，默认值为1
            getattr(self.env2, 'contrast', 1),
            choice
        ]

        # 保存到CSV文件中
        self.save_to_csv()

    def save_to_csv(self):
        # Specify the name of the CSV file
        filename = 'user_choices_16.csv'

        # Specify the field names of the CSV file
        fieldnames = ['Index','Scene', 'Env1 Color', 'Env2 Color', 'Env1 Brightness', 'Env2 Brightness',
                      'Env1 Texture', 'Env2 Texture', 'Probe1 Color', 'Probe2 Color',
                      'Probe1 Brightness', 'Probe2 Brightness', 'Probe1 Contrast', 'Probe2 Contrast', 'Choice']

        # Check if the file exists, if not create it and add a header line
        try:
            with open(filename, 'x', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
        except FileExistsError:
            pass

        # Determine the next serial number by reading the existing file
        serial_number = sum(1 for line in open(filename))  # Add one for each existing line
        
        # Open the file and append data to it
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([serial_number] + self.choice_data)


    def finish(self):
        top = tk.Toplevel(self)
        top.geometry("450x180")  # 调整对话框大小
        # 获取屏幕宽度和高度
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        # 计算对话框的位置
        x = (screen_width/2) - (300/2)
        y = (screen_height/2) - (120/2)

        top.geometry(f"+{int(x)}+{int(y)}")          # 设置对话框的位置在中间
        top.title("💜")
        label = tk.Label(top, text="\n\n Thank you for your participation! 💐 \n", font=('Times New Roman', 20))  # Label
        label.pack()
        button = tk.Button(top, text="OK", command=top.destroy, padx=10, pady=5)
        button.pack()

        
class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.title("🎨 Color Brightness Perception") # 设置窗口的标题
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)    # 创建新的帧实例
        if self._frame is not None:    # 如果当前帧存在，则销毁它
            self._frame.destroy()
        self._frame = new_frame      # 更新当前帧引用为新帧
        self._frame.pack()       # 打包新帧使其可见
       
        
if __name__ == "__main__":
    app = GUI()     # 创建GUI对象
    app.mainloop()   #启动主循环


# In[ ]:




