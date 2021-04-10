import pygame
import time
from socket import *

pygame.init()
s = socket(AF_INET, SOCK_STREAM)
s.bind(("", 3000))
s.listen(1)
print('connect waiting.....')

conn, addr = s.accept()
print('connected by', addr)

i=1
while 1:
    data = (conn.recv(1024)).decode()
    data = data[-1:]
    if not data:
        break

    if(i==1):
      if(data=='3'):
        pygame.mixer.music.load("red.wav")
        pygame.mixer.music.play()
        time.sleep(5)
      elif(data=='4'):
        pygame.mixer.music.load("green.wav")
        pygame.mixer.music.play()
        time.sleep(3)
      elif(data=='2'):
        pygame.mixer.music.load("blink.wav")
        pygame.mixer.music.play()
        time.sleep(3)
      elif(data=='1'):
        pygame.mixer.music.load("none.wav")
        pygame.mixer.music.play()
        time.sleep(3)

    elif(i>1):
      if(dataa!=data):
              if(data=='3'):
                pygame.mixer.music.load("red.wav")
                pygame.mixer.music.play()
                time.sleep(5)
              elif(data=='4'):
                pygame.mixer.music.load("green.wav")
                pygame.mixer.music.play()
                time.sleep(3)
              elif(data=='2'):
                pygame.mixer.music.load("blink.wav")
                pygame.mixer.music.play()
                time.sleep(3)
              elif(data=='1'):
                pygame.mixer.music.load("none.wav")
                pygame.mixer.music.play()
                time.sleep(3)

    dataa = data
    print(data)
    i+=1

conn.close()

