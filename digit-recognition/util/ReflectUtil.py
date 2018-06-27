# -*- coding: utf-8 -*-
"""
@Time: 2018/6/16 0016
@Author: ll
@File: ReflectUtil
@desc:
"""
import sys


def import_module(module_name):
    """
    导入模块
    :param module_name:模块全称
    :return: 模块对象
    """
    m = __import__(module_name)

    module_list = module_name.split('.')
    module_list.pop(0)
    for sub_module in  module_list:
        m = getattr(m, sub_module)
    return m


def import_class(class_name):
    segmentation_position = class_name.rindex('.')
    if segmentation_position != -1:
        m = import_module(class_name[0:segmentation_position])
        return getattr(m, class_name[segmentation_position+1:])
    else:
        return getattr(sys.modules['__main__'], class_name)