# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Welcome to the Workshop

# myVar = 10

from datetime import date
import numpy as np
import pandas as pd





myArr = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(date.today())

print(myArr)

northwestern = {
    "names": ["Seth", "Marco", "Isabelle"],
    "ID": [10,30,50],
    "Street Num": [109,800,450]
}


myDF = pd.DataFrame(northwestern, )

print(myDF)

"""decade = np.linspace(1,100,10)
spaced = np.arange(100,-1,-10)

print(spaced.max())

print(spaced)

print(spaced.tolist())
"""

class Student:

    #id = 1

    def __init__(self, age, name):
        print(f"Welcome to School {name}")
        self.age = age
        self.name = name
        #self.printID()

    def printID(self):
        print(f"Name: {self.name}, Age: {self.age}")

    def printIDandSome(self):
        print("Some")
        self.printID()




"""Marco = Student(22, "Marco")
Seth = Student(20, "Seth")
Jim = Student(name="Jim", age=21)

School = [Marco, Seth, Jim]

#for person in School:
#    person.printID()

Seth.printIDandSome()
"""

none = 0


"""def printName(name="No one"):
    print(f"Hello {name} Welcome")

def reverseName(name):
    reversed = name[0::-1]

    return reversed


myFile = open('testIn.txt','r')
myLog = open('testOut.txt','w')
myList = []

for line in myFile:
    myList.append(line)
    myLog.write(f"Log: {line}")


"""






'''
myVar2 = 0.25
myFrac = 1/5

myStr = "Welcome to the Zoom"
myStr2 = 'Hello'

myBoolean = False

myVar3 = None

print(myStr, myFrac)

myList = [90, 100, 4, -10]

myDictionary = {"Seth": 20, "Marco": 22, "Isabelle": 19}
myNumDictionary = {1: "First", 2: "Second"}

print(myDictionary["Marco"])

mySet = set()
myList.append("CompE")
myList.append("CS")
myList.append("EE")
myList.append("CS")

demo = 0

seth = ("Seth",20,"CompE",True)
sethList = ["Seth",20,"CompE",True]

print(sethList[-2])
'''
"""
nums = [10,20,30,40,50,80,15,45,67]
emptyList = []

for i in nums:
    print(i)
    print(i + 100)

    if i % 5 == 0:
        emptyList.append(i)

count = 10
while count > 0:
    print("Count",count)
    count -= 1

    if count == 5:
        

    print("Count #2", count)






print(nums)
print(emptyList)"""

