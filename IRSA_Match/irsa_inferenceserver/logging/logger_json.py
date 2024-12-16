# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:30:24 2020

@author: DYP
"""

import json
import time


class Log_json:
    def __init__(
            self,
            save_path: str = None,
            isprint: bool = True
    ):
        self.save_path = save_path

        self.log_data = dict()

        self.log_data['jobstate'] = 'running'
        self.log_data['Completed'] = 0
        self.log_data['nowstep'] = 1
        self.log_data['maxstep'] = 1
        self.log_data['log'] = []
        self.isprint = isprint
        self.save_log()

    def set_nowstep(self, now_step):
        self.log_data['nowstep'] = now_step
        self.save_log()
        if self.isprint:
            print('Step', now_step, '/', self.log_data['maxstep'])

    def set_maxstep(self, max_step):
        self.log_data['maxstep'] = max_step
        self.save_log()

    def set_process(self, nowiter: int):
        self.log_data['Completed'] = nowiter
        self.save_log()
        if self.isprint:
            print('Completedr', nowiter)

    def set_logstep(self, line_str: str):
        localtime = time.asctime(time.localtime(time.time()))
        line_str = localtime + ' : ' + line_str
        self.log_data['log'].append(line_str)
        self.save_log()
        if self.isprint:
            print(line_str)

    def set_state(self, state: str):
        print(state)
        if state not in ['error', 'running', 'finish']:
            raise ValueError('Not correct merge type `{}`.'.format(state))
        self.log_data['jobstate'] = state
        self.save_log()

    def save_log(self):
        try:
            with open(self.save_path, "w", encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, sort_keys=True, ensure_ascii=False)
        except IOError:
            print('Error:没有找到 文件或读取文件失败')


if __name__ == "__main__":
    Log_json('./log.json')


