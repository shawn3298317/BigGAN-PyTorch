import json
import multiprocessing
import subprocess
from multiprocessing import Process

import dateutil.parser
import numpy as np

import gpustat


def get_power_reading():
    out = subprocess.Popen(['sudo', '/home/software/perftools/0.1/bin/satori-ipmitool'],
                           stdout=subprocess.PIPE)
    out = subprocess.Popen(['grep', 'Instantaneous power reading'], stdin=out.stdout, stdout=subprocess.PIPE)
    out = subprocess.run(['awk', '{print $4}'], stdin=out.stdout, stdout=subprocess.PIPE)
    if out.returncode == 0:
        return out.stdout.decode().strip()


class Profile:
    def __init__(self, outfile=None):
        self.outfile = outfile
        self.manager = multiprocessing.Manager()
        self.data = self.manager.list()

    @staticmethod
    def record_power(data, outfile=None):
        while True:
            q = gpustat.new_query().jsonify()
            q['query_time'] = q['query_time'].isoformat()
            data.append({
                **q,
                'power': get_power_reading(),
            })

    def __enter__(self):
        self.p = Process(target=self.record_power,
                         args=(self.data,),
                         kwargs={'outfile': self.outfile})
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.terminate()
        self.p.join()
        if self.outfile is not None:
            with open(self.outfile, 'w') as f:
                json.dump(list(self.data), f)
            print(f'Wrote output to {self.outfile}')


class ProfileRun:

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
        return cls(data)

    def query_time(self, index):
        return dateutil.parser.parse(self.data[index]['query_time'])

    def power(self, index):
        return int(self.data[index]['power'])

    def _gpus(self, index):
        return self.data[index]['gpus']

    def _gpus_stats(self, index, name):
        return [d[name] for d in self._gpus(index)]

    def gpu_powers(self, index):
        return self._gpus_stats(index, 'power.draw')

    def gpu_temps(self, index):
        return self._gpus_stats(index, 'temperature.gpu')

    def gpu_utils(self, index):
        return self._gpus_stats(index, 'utilization.gpu')

    def gpu_mems(self, index):
        return self._gpus_stats(index, 'memory.used')

    def gpu_temp(self, index, gpu_index=None):
        out = self.gpu_temps(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_util(self, index, gpu_index=None):
        out = self.gpu_utils(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_mem(self, index, gpu_index=None):
        out = self.gpu_mems(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def gpu_power(self, index, gpu_index=None):
        out = self.gpu_powers(index)
        return np.mean(out) if gpu_index is None else out[gpu_index]

    def total_power(self):
        return sum([self.power(i) for i in range(len(self))]) / self.total_time()

    def total_gpu_power(self, gpu_index=None):
        return np.sum(self.gpu_power_profile(gpu_index=gpu_index)) / self.total_time()

    def average_power(self):
        return np.mean([self.power(i) for i in range(len(self))])

    def time_profile(self):
        return [(self.query_time(i) - self.query_time(0)).total_seconds() for i in range(len(self))]

    def total_time(self):
        return self.time_profile()[-1]

    def _profile(self, name):
        f = getattr(self, name)
        return [f(i) for i in range(len(self))]

    def _gprofile(self, name, gpu_index):
        f = getattr(self, name)
        return [f(i, gpu_index=gpu_index) for i in range(len(self))]

    def power_profile(self):
        return self._profile('power')

    def gpu_power_profile(self, gpu_index=None):
        return self._gprofile('gpu_power', gpu_index)

    def gpu_util_profile(self, gpu_index=None):
        return self._gprofile('gpu_util', gpu_index)

    def gpu_mem_profile(self, gpu_index=None):
        return self._gprofile('gpu_mem', gpu_index)

    def gpu_temp_profile(self, gpu_index=None):
        return self._gprofile('gpu_temp', gpu_index)

    def __len__(self):
        return len(self.data)

    def print_stats(self, gpu_index=None):
        print(f'Total time:      {self.total_time()}')
        print(f'Total Power:     {self.total_power() / (60 * 60)} Wh')
        print(f'Total GPU Power: {self.total_gpu_power(gpu_index=gpu_index) /(60 * 60)} Wh')
