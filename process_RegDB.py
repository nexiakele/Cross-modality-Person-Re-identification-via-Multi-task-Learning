# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:51:38 2020

@author: Dell
"""

import glob as gb
import os
def getFlist(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            p =  root + '\\' + f
            all_files.append(p)
    return all_files

if __name__ == '__main__':
    resDir = 'E:\数据库\RGB-T\RegDB'
    flist = getFlist(resDir)
#    print(len(flist))