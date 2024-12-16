import os
import sys
import logging
from logging import handlers
import threading


# class LogStream:
#     def __init__(self):
#         self.buffer = ''

#     def write(self, message):
#         self.buffer += message
#         if '\n' in message:
#             lines = self.buffer.split('\n')
#             for line in lines[:-1]:
#                 logging.info(line)
#             self.buffer = lines[-1]

#     def flush(self):
#         if self.buffer:
#             logging.info(self.buffer)
#             self.buffer = ''



class IRSALogger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename, level='info', backCount=360, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.logger = logging.getLogger(filename)
        self.format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        self.sh = logging.StreamHandler()#往屏幕上输出
        self.sh.setFormatter(self.format_str) #设置屏幕上显示的格式
        self.th = handlers.RotatingFileHandler(filename=filename, maxBytes=1024*1024*1024, backupCount=backCount, encoding='utf-8')
        # self.th = handlers.TimedRotatingFileHandler(filename=filename,when=when, backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        self.th.setFormatter(self.format_str)#设置文件里写入的格式
        self.logger.addHandler(self.sh) #把对象加到logger里
        self.logger.addHandler(self.th)
        
        self.buffer = ''
    
    def clearhandler(self):
        # self.logger.removeHandler(self.sh)
        self.logger.removeHandler(self.th)

    
    def write(self, message):
        
        self.buffer += message
        if '\n' in message:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                self.logger.info(line)
            self.buffer = lines[-1]
    
    def flush(self):
        if self.buffer:
            self.logger.info(self.buffer)
            self.buffer = ''


# class ThreadLogger:
#     def __init__(self, logger):
#         self.logger = logger

#     def write(self, message):
#         # 忽略换行符，因为logging会自动添加
#         if message != '\n':
#             self.logger.info(message)

#     def flush(self):
#         # 对于兼容sys.stdout的类，这个方法是必需的
#         pass


# thread_local = threading.local()

# def get_thread_logger(logpath):
#     os.makedirs(os.path.dirname(logpath), exist_ok=True)
#     if not hasattr(thread_local, 'logger'):
#         # 为每个线程创建一个独立的日志处理器
#         thread_logger = logging.getLogger(f"thread_{threading.get_ident()}")
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#         th = handlers.WatchedFileHandler(filename=logpath)#往文件里写入#指定间隔时间自动生成文件的处理器

#         handler.setFormatter(formatter)
#         th.setFormatter(formatter)#设置文件里写入的格式
#         thread_logger.addHandler(handler)
#         thread_logger.addHandler(th)
#         thread_logger.setLevel(logging.INFO)

#         # 将自定义的文件类实例分配给线程的logger属性
#         thread_local.logger = ThreadLogger(thread_logger)
#     return thread_local.logger