#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides methods for accessing the interface eAE API.

"""
import http
import json

from http import client
from uuid import uuid4

__author__ = "Axel Oehmichen"
__copyright__ = "Copyright 2017, Axel Oehmichen"
__credits__ = []
__license__ = "Apache 2"
__version__ = "0.1"
__maintainer__ = "Axel Oehmichen"
__email__ = "ao1011@imperial.ac.uk"
__status__ = "Dev"

__all__ = ['eAE']


class eAE(object):

    def __init__(self, interface_ip, interface_port):
        self.interface_ip = str(interface_ip)
        self.interface_port = int(interface_port)
        self.connection = http.client.HTTPSConnection(self.interface_ip, self.interface_port)

    #
    def __str__(self):
        return "\rThe interface ip is set to: {0}\r The interface port is set to: {1}".format(self.interface_ip,
                                                                                            self.interface_port)

    def is_eae_alive(self):
        """Retrieve the status of the eAE"""
        self.connection.request('GET', '/interfaceEAE/utilities/isAlive')
        res = self.connection.getresponse()
        return int(res.read())

    def retrieve_clusters(self):
        """Retrieve the list of all available clusters"""
        self.connection.request('GET', '/interfaceEAE/EAEManagement/retrieveClusters')
        res = self.connection.getresponse()
        str_response = res.read().decode('utf-8')
        clusters = json.loads(str_response)
        return clusters

    def submit_jobs(self, parameters_set, cluster, computation_type, main_file ,host_ip, ssh_port=22,):
        """Submit jobs to the eAE backend
        
        This method is called when a specific task needs to be deployed on a cluster.
        """
        #TODO: to be continued
        uuid = uuid4()
        zip_file = "{0}.zip".format(uuid)
        configs = parameters_set
        data = dict(id=uuid, host_ip=host_ip, ssh_port=ssh_port, zip=zip_file, configs=configs, cluster=cluster,
                    clusterType=computation_type, mainScriptExport=main_file)
        self.connection.request('POST', '/interfaceEAE/OpenLava/submitJob', data)
        return


def test_methods():
    ## Setting up the connection to interface
    ip = "interfaceeae.doc.ic.ac.uk"
    port = 443
    eae = eAE(ip, port)

    ## Testing if the interface is Alive
    is_alive = eae.is_eae_alive()
    print(is_alive)

    ## We retieve the list of Clusters
    clusters = eae.retrieve_clusters()
    print(clusters)

    ## We submit a dummy job
    job = eae.submit_jobs()
    print(job)

if __name__ == '__main__':
    test_methods()

