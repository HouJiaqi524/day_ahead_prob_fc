import os
from traceback import format_exc
from typing import Optional, Tuple

from numpy import nan
from pandas import DataFrame, concat, Timestamp, errors

from global_variable_manager.global_variable import INPUT_CONTENT_CODE_TYPE, INDEX_COL, UNIT_ID, ID_COL
from logger.logger import Logger


class FileInReader(object):

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
        for root, dirs, files in os.walk(in_path):
            for file in files:
                if file.split('.')[0] in list(var2atr.keys()):
                    block_name = file.split('.')[0]
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding=INPUT_CONTENT_CODE_TYPE) as f:
                            self.contents = f.readlines()
                            setattr(self, block_name, DataFrame(self.contents))
                            print(file_path, '解析完毕！')
                    except Exception as e:
                        print(e)
                        try:
                            with open(file_path, 'r', encoding='gbk') as f:
                                self.contents = f.readlines()
                                
                                setattr(self, self.var2atr[block_name], DataFrame(self.contents))
                                print(file_path, '解析完毕！')
                        except Exception as e:
                            print(e)
                            self.file_parse_debug_logger.error('Only support file parsing in GBK or UTF-8 format')
                            self.file_parse_debug_logger.error(format_exc())




                    self.file_parse_debug_logger.info('Parsing chunks')
                    for item in self.contents:
                        if item.find('<!') >= 0:
                            self.header = item
                            break

                    self._parse_blocks(block_name)
                    self.file_parse_debug_logger.info('Data read complete')
                    del self.contents

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
            # self._parse(block_name, temp)
            setattr(self, self.var2atr[block_name], temp)

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