#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorlayer as tl
from tensorlayer.db import TensorDB
import shutil
from eAE import eAE


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
    db = TensorDB(ip='146.169.33.34', port=27020, db_name='TransferGan', user_name='akara', password='DSIGPUfour', studyID="MNIST")
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
    ip = "interfaceeae.doc.ic.ac.uk"
    port = 443
    eae = eAE(ip, port)

    # Testing if the interface is Alive
    is_alive = eae.is_eae_alive()
    if(is_alive != 200){
        raise Exception
    }

    args = [
        "job_id database_meta", # seperate each arguments using space
        "job_id database_meta",
        "job_id database_meta"
    ]

    # We submit a dummy job
    parameters_set = "\n".join(args)
    cluster = "python_small"
    computation_type = "Python"
    main_file = "/home/akara/Workspace/tl_paper/tutorial_tensordb_cv_mnist_worker.py"
    data_files = ['/home/akara/Workspace/tl_paper/tensorlayer', '/home/akara/Workspace/tl_paper/tutorial_tensordb_atari_pong_trainer.py']
    host_ip = "dsihuaweiroom.doc.ic.ac.uk"
    ssh_port = "22"
    job = eae.submit_jobs(parameters_set, cluster, computation_type, main_file, data_files, host_ip, ssh_port)
    print(job)


if __name__ == '__main__':
    main()
