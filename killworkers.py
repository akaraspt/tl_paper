import os

PASSWORD='dsigdo2016!'
FILE_LOCATION='worker.py'
TARGET_LOCATION='/home/dsigdo/Workspace'

os.system("sshpass -p {} ssh -t dsigdo@gdo01.doc.ic.ac.uk 'killall python'".format(PASSWORD))
