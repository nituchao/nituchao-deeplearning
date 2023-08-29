#!/usr/bin/python
# -*- coding: UTF-8 -*-

class Employee:
    '所有员工的基类'
    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):
        print("Name : ", self.name, ", Salary: ", self.salary)

# 创建Employee类的第一个对象
emp1 = Employee("liang", "2000")
# 创建Employee类的第二个对象
emp2 = Employee("zhang", "5000")

emp1.displayEmployee()
emp2.displayEmployee()
print("Total Employee %d" % Employee.empCount)

# 为实例，添加一个'age'属性
emp1.age = 7 

print("Name: ", emp1.name, ", Age: ", emp1.age)

print(getattr(emp1, 'age'))
print(emp1.__dict__)
print(emp1.__doc__)
print(emp1.__module__)


