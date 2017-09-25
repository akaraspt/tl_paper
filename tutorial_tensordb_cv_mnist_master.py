#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorlayer as tl
from tensorlayer.db import TensorDB
import shutil
from eAE.eAE import eAE


def create_mnist_dataset(db):
    data, f_id = db.find_one_params(args={'type': 'mnist_dataset'})
    # If cannot find MNIST dataset in TensorDB
    if not data:
        # Download and upload MNIST dataset to TensorDB
        X_train, y_train, X_val, y_val, X_test, y_test = \
            tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        f_id = db.save_params(
            [X_train, y_train, X_val, y_val, X_test, y_test],
            args={'type': 'mnist_dataset'}
        )
        shutil.rmtree('./data/mnist')


def create_jobs(db, job_name, models_dict):
    # job = db.find_one_job(args={'job_name': job_name})
    # if not job:
    #     job_idx = 1
    #     for model, params_dict in models_dict.iteritems():
    #         n_jobs = len(params_dict.itervalues().next())
    #         for j in range(n_jobs):
    #             job_dict = {'model': model, 'job_name': job_name, 'job_id': job_idx}
    #             for k, v in params_dict.iteritems():
    #                 job_dict.update({k: v[j]})
    #             db.save_job(args=job_dict)
    #             job_idx += 1
    # else:
    #     print("You have already submitted this job.")
    for model, params_dict in models_dict.iteritems():
        n_jobs = len(params_dict.itervalues().next())
        for j in range(n_jobs):
            job_dict = {'model': model}
            for k, v in params_dict.iteritems():
                job_dict.update({k: v[j]})
            db.save_job(args=job_dict)


def start_workers(db):
    job_ids = []
    for job in db.get_all_jobs():
        job_ids.append(str(job['_id']))

    # Check how many available workers
    workers = ['node01', 'node02', 'node03', 'node04', 'node05']

    def submit_job(node_name, job_id):
        print('Assign job: {} to {}'.format(job_id, node_name))
        worker(job_id)

    # Submit jobs to all workers
    submit_job(workers[0], job_ids[0])
    submit_job(workers[2], job_ids[2])
    submit_job(workers[4], job_ids[4])


def main():
    # This is to initialize the connection to your MondonDB server
    # Note: make sure your MongoDB is reachable before changing this line
    db = TensorDB(ip='IP_ADDRESS_OR_YOUR_MONGODB', port=27017, db_name='DATABASE_NAME', user_name=None, password=None, studyID='ANY_ID (e.g., mnist)')

    create_mnist_dataset(db=db)
    create_jobs(db=db, job_name="cv_mnist", models_dict={
        "cnn": {
            "lr": [0.01, 0.001, 0.001],
            "n_cnn_layers": [1, 2, 2],
            "n_filters": [64, 128, 256],
            "n_epochs": [10, 10, 10],
        },
        "mlp": {
            "lr": [0.05, 0.0001],
            "n_layers": [1, 2],
            "n_epochs": [10, 10],
        }
    })

    # Setting up the connection to interface
    ip = "IP_ADDRESS_OF_EAE (e.g., interfaceeae.doc.ic.ac.uk)"
    port = 443
    eae = eAE(ip, port)

    # Testing if the interface is Alive
    is_alive = eae.is_eae_alive()
    if is_alive != 200:
        raise Exception("!!!")

    # Get all jobs
    jobs = db.get_all_jobs()
    args = [str(j['_id']) for j in jobs]

    # We submit a dummy job
    parameters_set = "\n".join(args)
    cluster = "NAME_OF_CLUSTER (e.g., gpu_dev)"
    computation_type = "COMPUTATION_TYPE (e.g., GPU)"
    main_file = "ABSOLUTE_PATH_TO_MAIN_FILE"
    data_files = ['ABSOLUTE_PATH_TO_DIRECTORY_OR_FILES_TO_BE_COPIED_TO_RUN_THE_MAIN_FILE']
    host_ip = "IP_ADDRESS_OF_HOST_MACHINE_RUNNING_THIS_SCRIPT"
    ssh_port = "SSH_PORT_OF_HOST_MACHINE"
    job = eae.submit_jobs(parameters_set, cluster, computation_type, main_file, data_files, host_ip, ssh_port)
    print(job)


if __name__ == '__main__':
    main()
