import os

PASSWORD='dsigdo2016!'
FILE_LOCATION='worker.py'
TARGET_LOCATION='/home/dsigdo/Workspace'

#===== Start one worker
# sshpass -p $PASSWORD scp worker.py dsigdo@gdo01.doc.ic.ac.uk:/home/dsigdo/Workspace
# sshpass -p $PASSWORD ssh -t dsigdo@gdo01.doc.ic.ac.uk 'python /home/dsigdo/Workspace/worker.py'

# os.system("sshpass -p {} scp {} dsigdo@gdo01.doc.ic.ac.uk:{}".format(PASSWORD, FILE_LOCATION, TARGET_LOCATION))
# os.system("sshpass -p {} ssh -t dsigdo@gdo01.doc.ic.ac.uk 'python3 {}/{} &'".format(PASSWORD, TARGET_LOCATION, FILE_LOCATION))

#===== Start multiple workers
NUMBER_WORKER = 10
for i in range(NUMBER_WORKER):
    ID = str(i).zfill(2)
    print("Start worker: " + ID)
    os.system("sshpass -p {} scp {} dsigdo@gdo{}.doc.ic.ac.uk:{}".format(PASSWORD, FILE_LOCATION, ID, TARGET_LOCATION))
    os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'python3 {}/{} &'".format(PASSWORD, ID, TARGET_LOCATION, FILE_LOCATION))
