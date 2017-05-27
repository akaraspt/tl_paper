from tensorlayer.db import TensorDB
from tensorlayer.db import JobStatus
import shutil
from eAE.eAE import eAE


def main():
    # db = TensorDB(ip='146.169.33.34', port=27020, db_name='DRL', user_name='tensorlayer', password='Tensorlayer123', studyID="20170524_1")
    db = TensorDB(ip='146.169.15.140', port=27017, db_name='DRL', user_name=None, password=None, studyID="1")

    # Create jobs
    n_jobs = 5
    for j in range(n_jobs):
        args = {
            "id": j,
            "name": "Deep Reinforcement Learning",
            "file": "tutorial_tensordb_atari_pong_generator.py",
            "args": "",
        }
        db.submit_job(args=args)

    # Setting up the connection to interface
    ip = "interfaceeae.doc.ic.ac.uk"
    port = 443
    eae = eAE(ip, port)

    # Testing if the interface is Alive
    is_alive = eae.is_eae_alive()
    if is_alive != 200:
        raise Exception("!!!")

    # Get all jobs
    jobs = db.get_jobs(status=JobStatus.WAITING)

    for j in jobs:
        # Start worker
        parameters_set = "--job_id={}".format(str(j["_id"]))
        cluster = "gpu"
        computation_type = "GPU"
        main_file = j["file"]
        data_files = ['tensorlayer']
        host_ip = "dsigpu2.ict-doc.ic.ac.uk"
        ssh_port = "22222"
        job = eae.submit_jobs(parameters_set, cluster, computation_type, main_file, data_files, host_ip, ssh_port)
        db.change_job_status(job_id=j["_id"], status=JobStatus.RUNNING)
        print(job)


if __name__ == "__main__":
    main()
