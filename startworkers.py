import os

PASSWORD='dsigdo2016!'
FILE_LOCATION='worker.py'
TARGET_LOCATION='/home/dsigdo/Workspace'

#===== Start one worker
# sshpass -p $PASSWORD scp worker.py dsigdo@gdo01.doc.ic.ac.uk:/home/dsigdo/Workspace
# sshpass -p $PASSWORD ssh -t dsigdo@gdo01.doc.ic.ac.uk 'python /home/dsigdo/Workspace/worker.py'

# os.system("sshpass -p {} scp {} dsigdo@gdo01.doc.ic.ac.uk:{}".format(PASSWORD, FILE_LOCATION, TARGET_LOCATION))
# os.system("sshpass -p {} ssh -t dsigdo@gdo01.doc.ic.ac.uk 'python3 {}/{} &'".format(PASSWORD, TARGET_LOCATION, FILE_LOCATION))

# os.system("sshpass -p {} scp {} haodong@155.198.188.11:".format(PASSWORD, FILE_LOCATION, ID))
# os.system("sshpass -p {} ssh -t haodong@155.198.188.11 'python3 {}/{} &'".format(PASSWORD, ID, TARGET_LOCATION, FILE_LOCATION))
# os.system("sshpass -p {} ssh -t haodong@155.198.188.11 'pip3 install pymongo gym numpy matplotlib scipy scikit-image'".format(PASSWORD))



# #===== Start multiple workers
# NUMBER_WORKER = 10
# for i in range(NUMBER_WORKER):
#     ID = str(i).zfill(2)
#     print("Start worker: " + ID)
#     os.system("sshpass -p {} scp {} dsigdo@gdo{}.doc.ic.ac.uk:{}".format(PASSWORD, FILE_LOCATION, ID, TARGET_LOCATION))
#     os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'python3 {}/{} &'".format(PASSWORD, ID, TARGET_LOCATION, FILE_LOCATION))
#     os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'pip3 install pymongo gym numpy matplotlib scipy scikit-image'".format(PASSWORD, ID))

# #===== Start multiple workers
NUMBER_WORKER = 32
for i in range(1, NUMBER_WORKER+1):
    ID = str(i).zfill(2)
    print("Start worker: " + ID)
    # os.system("sshpass -p {} scp {} dsigdo@gdo{}.doc.ic.ac.uk:{}".format(PASSWORD, FILE_LOCATION, ID, TARGET_LOCATION))
    # os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'python3 {}/{} &'".format(PASSWORD, ID, TARGET_LOCATION, FILE_LOCATION))
    # os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'sudo apt-get install cmake'".format(PASSWORD, ID))
    # os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'pip install gym[atari]'".format(PASSWORD, ID))

    # os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'sudo dpkg --configure -a'".format(PASSWORD, ID))
    # os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'sudo apt-get install cmake swig'".format(PASSWORD, ID))
    # kill python
    os.system("sshpass -p {} ssh -t dsigdo@gdo{}.doc.ic.ac.uk 'killall Python'".format(PASSWORD, ID))
