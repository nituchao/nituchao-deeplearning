#!/usr/bin/python
# -*- coding: UTF-8 -*-

class GrandFather:
    def __init__(self):
        print('GrandFather')

class Father(GrandFather):
    def __init__(self):
        print('Father')
    
class Son(Father):
    def __init__(self):
        print('Son')
        super(Son, self).__init__()

        print('准备绕过爹')
        super(Father, self).__init__()

son = Son()