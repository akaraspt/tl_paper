from tensorlayer.db import TensorDB
from tensorlayer.db import JobStatus

db = TensorDB(ip='146.169.15.140', port=27017, db_name='DRL', user_name=None, password=None, studyID="1")

# Terminate running jobs
jobs = db.get_jobs(status=JobStatus.RUNNING)
for j in jobs:
    print db.change_job_status(job_id=j["_id"], status=JobStatus.TERMINATED)
