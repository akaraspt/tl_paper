# import os
# os.system("echo ======================================================")
# os.system("echo $(logname)")
# os.system("ifconfig | grep inet")

from tensorlayer.db import TensorDB
import time

while True: # scan job from dataset
    # if job exist, start program

    # wait 0.5 second
    time.sleep(0.5)
