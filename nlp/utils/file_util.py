#!/usr/bin/python3
"""
@File:   file_util.py
@Author: yczou
@Time:   2020-10-19 15:57:05
@Desc:   存放文件处理相关的方法
"""
import hashlib
import os
import traceback

import chardet

DEFAULT_ENCODING='utf-8'

def write_lines(lines, file_name):
    """
    把字符串数组写入到指定文件里
    :param lines:
    :param file_name:
    :return:
    """
    with open(file_name, 'w', encoding=DEFAULT_ENCODING) as text_file:
        text_file.write('\n'.join(lines))


def append_line(line, file_name):
    """
    往指定的文件追加一行
    :param line:
    :param file_name:
    :return:
    """
    with open(file_name, 'a', encoding=DEFAULT_ENCODING) as text_file:
        text_file.write(line + '\n')


def append_lines(lines, file_name):
    """
    往指定的文件追加多行
    :param lines:
    :param file_name:
    :return:
    """
    with open(file_name, 'a', encoding=DEFAULT_ENCODING) as text_file:
        text_file.write('\n'.join(lines))


def read_whole_file(file_name):
    """
    读取文本文件的所有内容
    :param file_name:
    :return:
    """
    if not os.path.exists(file_name):
        return ''

    with open(file_name, 'rb') as text_file:
        content = text_file.read()
        encoding = chardet.detect(content)['encoding']
        return content.decode(encoding)


def read_lines(file_name):
    """
    按行读取文本文件并返回
    :param file_name:
    :return:
    """
    content = read_whole_file(file_name).replace('\r', '')
    return content.split('\n')


def calc_file_md5(file_name):
    with open(file_name, 'rb') as reader:
        content = reader.read()

    return hashlib.md5(content).hexdigest()


def get_text_file_encoding(file_name):
    if not os.path.exists(file_name):
        return ''

    with open(file_name, 'rb') as text_file:
        content = text_file.read()
        return chardet.detect(content)['encoding']


def move_file_2_dic(source_file, dest_dic):
    """
    移动文件到目标文件夹，若目标文件夹下有同名文件直接覆盖
    :param source_file:
    :param dest_dic:
    :return:
    """
    target_file = os.path.join(dest_dic, source_file[source_file.rfind('/') + 1:])

    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(source_file, target_file)


def get_file_suffix(file_name):
    """
    获取文件后缀
    :param file_name:
    :return:
    """
    if file_name is None:
        return ''

    strs = file_name.split('.')
    if len(strs) > 1:
        return strs[len(strs) - 1]

    return ''


def delete_file(file_path):
    """
    删除文件（不会抛异常）
    :param file_path:
    :return:
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        print('delete_file: {} error!! {}'.format(file_path, traceback.format_exc()))
