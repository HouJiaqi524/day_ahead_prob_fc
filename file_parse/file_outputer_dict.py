import os
import logging
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed


class FileOutManager:
    def __init__(self, out_path: str, data, block_name_dict: dict, header, args):
        """
        :param out_path: 输出路径
        :param data: 待写出数据，格式为 {block_name: [[row1], [row2], ...]}
        """
        self.out_path = out_path
        self.data = data
        self.logger = logging.getLogger('DataWriter')
        self.header = header
        self.block_name_dict  = block_name_dict
        self.args = args
        
        self.write_all()

    def write_all(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        # 多线程写入（适合 I/O 密集型任务）
        with ThreadPoolExecutor(self.args.cpu_thread) as executor:
            futures = []
            for attr_name, block_name in self.block_name_dict.items():
                futures.append(
                    executor.submit(self.__write_block, block_name, getattr(self.data,attr_name))
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error writing file: {e}")

    def __write_block(self, block_name: str, content: List[List]):
        out_file_name = os.path.join(self.out_path, f"{block_name}.txt")
        
        # 如果文件已存在，先删除
        if os.path.exists(out_file_name):
            try:
                os.remove(out_file_name)
            except Exception as e:
                self.logger.warning(f"Failed to remove existing file {out_file_name}: {e}")

        # 构造 header 和 body
        header = f"<{block_name}>\n"
        footer = f"</{block_name}>\n\n"

        if block_name ==  'Variance' or 'SceneProb' in block_name:
            content = [" ".join(map(str, row)) for row in content]

        with open(out_file_name, mode='w', encoding='utf-8') as f:
            f.write(self.header)
            f.write(header)
            f.write("\n".join(content) + "\n")
            # for row in content:
            #     f.write(" ".join(map(str, row)) + "\n")
            f.write(footer)

        self.logger.info(f"Block <{block_name}> has been written.")
        
        
# import numpy as np

# def write_dict_arrays_to_file(data_dict, output_file):
#     """
#     将 dict[unit_id] = array 的结构转换为带 unit_id 标识的嵌套 list，
#     并一次性写出为文本文件，每行格式如下：
#         'unit_001' 1 2
#         'unit_001' 3 4
#         ...
    
#     :param data_dict: 输入字典，格式为 {unit_id: np.ndarray}
#     :param output_file: 输出文件路径
#     """
#     # 1️⃣ 构造所有行内容（带 unit_id）
#     lines = []
#     for unit_id, arr in data_dict.items():
#         for row in arr:
#             line = f"{unit_id} " + " ".join(map(str, row))
#             lines.append(line)

#     # 2️⃣ 一次性写入文件（大幅减少 I/O 次数）
#     with open(output_file, "w") as f:
#         f.write("\n".join(lines) + "\n")
    
#     print(f"已成功写出 {len(lines)} 行到文件：{output_file}")