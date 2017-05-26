import os

PASSWORD='dsigdo2016!'
FILE_LOCATION='worker.py'
TARGET_LOCATION='/home/dsigdo/Workspace'


# #===== Start multiple workers
NUMBER_WORKER = 32
for i in range(1, NUMBER_WORKER+1):
    ID = str(i).zfill(2)
    print("Kill worker: " + ID)
    os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'killall Python'".format(PASSWORD, ID))
    os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'killall python'".format(PASSWORD, ID))
