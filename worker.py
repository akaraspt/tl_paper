# import os
# os.system("echo ======================================================")
# os.system("echo $(logname)")
# os.system("ifconfig | grep inet")

from tensorlayer.db import TensorDB
import time, os

TARGET_LOCATION='/home/dsigdo/Workspace'

while True: # scan job from dataset
    # if job exist, start program
    # os.system("python3 {}/{}".format(TARGET_LOCATION, xx))
    # wait 0.5 second
    time.sleep(0.5)
