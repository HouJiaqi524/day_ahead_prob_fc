import os
import shutil
from datetime import datetime
from os import remove
from pandas import Series, DataFrame
from global_variable_manager.global_variable import OUTPUT_CONTENT_CODE_TYPE
from logger.logger import Logger


class FileOutManager(object):
    def __init__(self, out_path, out_content, block_name: str = None, block_name_dict: dict = None, header: str = None):
        self.file_write_debug_logger = Logger('DataWriter')
        self.out_path: str = out_path
        self.out_content = out_content
        self.block_name_dict = block_name_dict
        self.block_name = block_name
        self.header = header
        self.out_content_one = None
        for atr_name in self.block_name_dict:
            self.__write_content(atr_name)

    def __write_content(self, atr_name: str):
        block_name = self.block_name_dict[atr_name]
        out_file_name = os.path.join(self.out_path, block_name+'.txt')
        try:
            os.remove(out_file_name)  # remove the original one
        except:
            pass
        with open(out_file_name, mode='a', encoding=OUTPUT_CONTENT_CODE_TYPE) as f:
            f.write(self.header)
            block_name_begin = f'<{block_name}>' + '\n'
            block_name_end = f'</{block_name}>' + '\n' + '\n'
            exec(f'self.out_content_one = self.out_content.{atr_name}.round(4).astype(str)')
            content_header, content_value = self.__generate_content(self.out_content_one)
            f.write(block_name_begin)
            f.write(content_header)
            f.write(content_value)
            f.write(block_name_end)
            self.file_write_debug_logger.info(f'<{block_name}>has been written')
        f.close()

    @staticmethod
    def __generate_content(out_content: Series or DataFrame):
        if isinstance(out_content, Series):
            content_header = '@  ' + "  ".join(out_content.index.tolist()) + '\n'
            content_value = '#  ' + "  ".join(out_content.values.tolist()) + '\n'
        else:
            content_header = "@  " + "  ".join(out_content.columns.tolist()) + '\n'
            content_value = ''
            for i in range(out_content.shape[0]):
                content_value += '#  ' + "  ".join(out_content.iloc[i, :].tolist()) + '\n'
        return content_header, content_value



