import os
from traceback import format_exc
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path

from numpy import nan
from pandas import DataFrame, concat, Timestamp, errors

from global_variable_manager.global_variable import INPUT_CONTENT_CODE_TYPE, INDEX_COL, UNIT_ID, ID_COL
from logger.logger import Logger


import mmap
import re
from typing import Tuple, List, Optional
import multiprocessing as mp
import pandas as pd

def _read_file_bytes(fp: Path, bn: str):
    """线程池任务：读取文件全部字节"""
    with open(fp, 'rb') as f:
        return bn, f.read()


from io import BytesIO

def parse_bytes_readlines_fast(file_bytes: bytes):
    """
    超快解析 bytes -> DataFrame（固定格式文件，内存充足）
    - header 行以 '@' 开头
    - 数据行以 '#' 开头
    - 列数固定，用空格分隔
    - 尝试 utf-8 解码，失败 fallback gbk
    """
    # 尝试 decode 整块文件
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = file_bytes.decode('gbk')

    header = None
    data = []

    for line in text.splitlines():
        if not line:  # 空行跳过
            continue
        first_char = line[0]
        if first_char == '@':
            header = line[1:].split()  # 不再 strip，split 自动去空格
        elif first_char == '#':
            data.append(line[1:].split())

    if header and data:
        return pd.DataFrame(data, columns=header)
    return None

def _parse_bytes_to_df(file_bytes: bytes):
    """进程池：解析 bytes -> DataFrame"""
    header = None
    data = []
    for raw_line in file_bytes.split(b'\n'):
        if not raw_line.strip():
            continue
        if raw_line.startswith(b'@'):
            try:
                header = raw_line[1:].strip().decode('utf-8').split()
            except:
                header = raw_line[1:].strip().decode('gbk').split()
        elif raw_line.startswith(b'#'):
            try:
                data.append(raw_line[1:].strip().decode('utf-8').split())
            except:
                data.append(raw_line[1:].strip().decode('gbk').split())

    if header and data:
        return pd.DataFrame(data, columns=header)
    return None


def parse_chunk(chunk: bytes) -> Tuple[Optional[List[str]], List[List]]:
    """
    解析一个文本块（bytes），返回：
    - header: 表头（只在第一个有 @ 的块中有效）
    - data: 数据行列表
    """
    pattern = re.compile(rb'#\s*(\S+)\s+(.*?)\s+\[(.*?)\]')
    
    header = None
    data = []
    
    # 按行处理
    for line in chunk.split(b'\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(b'@'):
            # 解析表头
            try:
                header = line[1:].strip().decode('utf-8').split()
            except:
                header = line[1:].strip().decode('gbk').split()
        elif line.startswith(b'#'):
            try:
                data_line = line[1:].strip().decode('utf-8').split()
            except:
                data_line = line[1:].strip().decode('gbk').split()
            data.append(data_line)
            # match = pattern.match(line)
            # if match:
            #     obj_name = match.group(1).decode('utf-8')
            #     chinese_name = match.group(2).decode('utf-8').strip()
            #     try:
            #         stations = list(map(int, match.group(3).split(b',')))
            #     except:
            #         stations = []  # 解析失败
                
    
    return header, data



def find_line_offsets(mm) -> List[int]:
    """找到所有换行符后的位置，用于按行切块"""
    offsets = []
    start = 0
    while True:
        pos = mm.find(b'\n', start)
        if pos == -1:
            break
        offsets.append(pos + 1)  # 下一行开始位置
        start = pos + 1
    return [0] + offsets  # 从文件开头开始


def parse_parallel_smart(filename: str, num_workers: int = 4):
    """
    并行解析大文件，安全处理行边界
    """
    with open(filename, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 1. 获取所有行的起始偏移
            offsets = find_line_offsets(mm)
            n_lines = len(offsets) - 1  # 最后一个 offset 是文件末
            if n_lines == 0:
                raise ValueError("文件为空")

            # 2. 分块：每块包含完整行
            chunk_size = (n_lines + num_workers - 1) // num_workers  # 向上取整
            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, n_lines)
                
                if start_idx >= n_lines:
                    break
                    
                start = offsets[start_idx]
                end = offsets[end_idx] if end_idx < len(offsets) else len(mm)
                chunk_data = mm[start:end]
                chunks.append(chunk_data)

    # 3. 并行解析
    with mp.Pool(num_workers) as pool:
        results = pool.map(parse_chunk, chunks)

    # 4. 合并结果
    final_data = []
    final_header = None
    
    for header, data in results:
        if header is not None and final_header is None:
            final_header = header
        final_data.extend(data)
    
    # 5. 构建 DataFrame
    if final_header is None:
        final_header = ['ObjectName', 'ChineseName', 'StationList']
        
    df = pd.DataFrame(final_data, columns=final_header)
    return df





class FileInReader(object):
    """
    多线程读取多个txt文件，增加读取效率
    """

    def __init__(self, in_path: str, var2atr: dict, var_type: dict, args=None):
        self.file_parse_debug_logger = Logger("FileParse")
        self.file_parse_debug_logger.info('Reading content of file FILE_IN')
        self.control_param_dict = {}  
        self.algo_param_dict = {}  
        self.no_parse_atr = []  
        self.no_data_atr = []  
        self.atr_list = list(var2atr.values())
        for name in self.atr_list:
            setattr(self, name, None)
        self.var2atr = var2atr
        self.var_type = var_type
        # for root, dirs, files in os.walk(in_path):
        #     for file in files:
        #         name_without_ext = file.split('.')[0]
        #         if name_without_ext in self.var2atr:
        #             file_path = os.path.join(root, file)
        #             df = parse_parallel_smart(file_path, num_workers=args.cpu_thread)
        #             setattr(self, self.var2atr[name_without_ext], df)
        
        """多线程解析所有匹配文件"""
        files_to_process = []
        # 扫描所有匹配的文件
        for root, dirs, files in os.walk(in_path):
            for file in files:
                name_without_ext = file.split('.')[0].split('_')[0]
                if name_without_ext in self.var2atr:
                    file_path = os.path.join(root, file)
                    files_to_process.append((file_path, name_without_ext))

        # 多线程处理
        results = {}
        with ThreadPoolExecutor(max_workers=args.cpu_thread) as t_pool, \
            ProcessPoolExecutor(max_workers=args.cpu_thread) as p_pool:

            # Step 1: 线程池读取
            read_futures = {
                t_pool.submit(_read_file_bytes, fp, bn): (fp, bn)
                for fp, bn in files_to_process
            }

            # Step 2: 丢进进程池解析
            parse_futures = {}
            for f in as_completed(read_futures):
                fp, bn = read_futures[f]
                try:
                    bn, file_bytes = f.result()
                    pf = p_pool.submit(parse_bytes_readlines_fast, file_bytes)
                    parse_futures[pf] = bn
                except Exception as e:
                    print(f"[读取失败] {fp} -> {e}")

            # Step 3: 解析完直接存类属性
            for f in as_completed(parse_futures):
                bn = parse_futures[f]
                try:
                    df = f.result()
                    if df is not None:
                        setattr(self, self.var2atr[bn], df)
                        print(f"[完成] {bn} -> 存到 {self.var2atr[bn]}")
                except Exception as e:
                    print(f"[解析失败] {bn} -> {e}")
        


        

    def _parse_blocks(self, block_name:str):
        def f(begin_idx, end_idx, block_name=block_name):
            block_lines = [line.strip() for line in self.contents[begin_idx + 1:end_idx] if line.strip()]
            try:
                columns = block_lines[0].split()
                data = [line.split() for line in block_lines[1:]]
                temp = DataFrame(data=data, columns=columns)
            except Exception as e:
                print(e)
                temp = DataFrame(columns=block_lines[0])
            temp = temp[temp['@'] == '#'].drop(['@'], axis=1)
            self._parse(block_name, temp)

        begin_idx, end_idx = self._loc_block(block_name)
        f(begin_idx, end_idx)

    def _parse(self, block_name: str, temp: DataFrame, data_type=int) -> None:
        self.file_parse_debug_logger.info(f'Parsing {block_name}')
        parse_func = {
            None: lambda x: x,
            'col': lambda x: self._parse_col(x, block_name, num_type=data_type),
            'table': lambda x: self._parse_table(x, block_name)
        }
        parse_result = parse_func[self.var_type[block_name]](temp)
        parse_result = parse_result.reset_index(drop=True)
        setattr(self, self.var2atr[block_name], parse_result)

    def _loc_block(self, block_name: str) -> Tuple[int, int]:
        begin_idx, end_idx = 0, 0
        # TODO: changed! f'<{block_name}>' to f'<{block_name}'
        if 'Feature' in block_name:
            count = 0
            begin_idxs = []
            end_idxs = []
            for idx, line in enumerate(self.contents):
                if f'<{block_name}' in line:
                    begin_idx = idx
                    begin_idxs.append(begin_idx)
                if f'</{block_name}' in line:
                    end_idx = idx
                    end_idxs.append(end_idx)
            return begin_idxs, end_idxs
        # TODO: changed! f'<{block_name}>' to f'<{block_name}'
        else:
            for idx, line in enumerate(self.contents):
                if f'<{block_name}>' in line:
                    begin_idx = idx
                if f'</{block_name}>' in line:
                    end_idx = idx
                    break
            return begin_idx, end_idx

    def _parse_col(self, temp: DataFrame, block_name: str, num_type: Optional[type] = None) -> Optional[DataFrame]:
        try:
            if temp.shape[0] == 0:
                self._log_no_data(block_name)
            else:
                for column in temp.columns:
                    if column == INDEX_COL:
                        temp[column] = temp[column].astype(str)
                        temp[column] = temp[column].apply(Timestamp)
                    elif column == UNIT_ID:
                        temp[column] = temp[column].astype(str)
                    elif column == ID_COL:
                        temp[column] = temp[column].astype(str)
                    else:
                        temp[column] = temp[column].replace('null', nan)
                        temp[column] = temp[column].replace('NULL', nan)
                        temp[column] = temp[column].replace('Null', nan).astype(float)
            return temp

        except errors.EmptyDataError:
            self._log_no_data(block_name)
            return None

        except Exception as e:
            self._log_parse_error(block_name, e)
            return None

    def _parse_table(self, temp: DataFrame, block_name: str, index_col: str = "Date") -> Optional[DataFrame]:
        """
        Parse DataFrame and convert it to a new DataFrame with a date column and several float columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            block_name (str): Name of the block corresponding to the DataFrame.
            index_col (str): Name of the column that contains dates. Default is 'Date'.

        Returns:
            pd.DataFrame or None: Parsed DataFrame. If parsing fails, returns None and logs error message.
        """
        try:
            if temp.shape[0] == 0:
                self._log_no_data(block_name)
            else:
                for column in temp.columns:
                    if column == INDEX_COL:
                        temp[column] = temp[column].astype(str)
                        temp[column] = temp[column].apply(Timestamp)
                    elif column == UNIT_ID:
                        temp[column] = temp[column].astype(str)
                    elif column == ID_COL:
                        temp[column] = temp[column].astype(str)
                    else:
                        temp[column] = temp[column].replace('null', nan)
                        temp[column] = temp[column].replace('NULL', nan)
                        temp[column] = temp[column].replace('Null', nan).astype(float)

            return temp

        except errors.EmptyDataError:
            self._log_no_data(block_name)
            return None

        except Exception as e:
            self._log_parse_error(block_name, e)
            return None

    def _log_no_data(self, block_name: str) -> None:
        """
        Log error message when the DataFrame passed to parse_table is empty.
        """
        self.file_parse_debug_logger.info(f"<{block_name}> DataFrame contains no data")
        self.no_data_atr.append(self.var2atr[block_name])

    def _log_parse_error(self, block_name: str, error: Exception) -> None:
        """
        Log error message when parsing of the DataFrame in parse_table fails.
        """
        self.file_parse_debug_logger.warning(
            f"<{block_name}> parsing error occurred; this information will not be used to ensure normal "
            f"operation of the algorithm"
        )
        self.file_parse_debug_logger.warning(error)
        self.no_parse_atr.append(self.var2atr[block_name])