'''
Author: houjq houjq@tsintergy.com.cn
Date: 2024-07-05 11:38:29
LastEditors: houjq houjq@tsintergy.com.cn
LastEditTime: 2024-07-09 09:48:54
FilePath: \jilin_day_ahead_power_prob_fc\file_parse\command_parse.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from os import remove, path
from re import search
from sys import argv
from traceback import print_exc

from argparse import ArgumentParser

from global_variable_manager.global_variable import IN_NAME, OUT_NAME, DEBUG_NAME, AUTH_NAME


def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
        print(path)

class CommandParse:
    def __init__(self):

        self.in_file_path = self.parse_path('/p:')# + IN_NAME
        if self.parse_path('/p:') == '':
            self.in_file_path = os.getcwd()
        else:
            Mkdir(self.parse_path('/p:'))
        self.out_file_path = self.parse_path('/e:')# + OUT_NAME
        if self.parse_path('/e:') == '':
            self.out_file_path = os.getcwd()
            # Mkdir(os.path.join(self.out_file_path, '_1'))
            # Mkdir(os.path.join(self.out_file_path, '_2'))
        else:
            Mkdir(self.parse_path('/e:'))
            # Mkdir(os.path.join(self.parse_path('/e:'), '_1'))
            # Mkdir(os.path.join(self.parse_path('/e:'), '_2'))
        self.debug_file_path = os.path.join(self.out_file_path, DEBUG_NAME)
        self.auth_file_path = self.parse_path('/c:')# + AUTH_NAME
        self.bar_file_path = os.path.join(self.out_file_path, 'process_bar.e')
        try:
            if path.exists(self.debug_file_path):
                remove(self.debug_file_path)
            if path.exists(self.bar_file_path):
                remove(self.bar_file_path)
        except FileNotFoundError:
            pass
        except PermissionError:
            print('not have permission to remove the debug.e')
        except Exception as e:
            print(f"An error occurred: {e}")
            print_exc()

    def parse_path(self, command_head: str) -> str:
        command = filter(lambda x: x.startswith(command_head), argv)
        _path = ''.join(command)
        _path = self.ensure_tailing_slash(_path).replace(r'\\', '/').replace('//', '/')
        if search(':', _path):
            idx = search(':', _path).end()
        else:
            idx = 0
        return _path[idx:]

    @staticmethod
    def ensure_tailing_slash(_path: str) -> str:
        return _path.rstrip('/') + '/' if _path else ''
    
    


def parse_args():
    parser = ArgumentParser(description="命令行参数解析")

    parser.add_argument('--cpu_thread', type=int, required=False, default=2,  help='使用核心数') 
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()
    return args
