from typing import TypedDict, Optional

class Person(TypedDict):
    name : str
    age : Optional[int]

new_person1 = {"name" : "rahul"}
new_person2 = {"name" : "rahul", "age" : 32}
new_person3 = {"name" : "rahul", "age" : "32"} # it will accept even though it is not in integer

person1 = Person(new_person1)
person2 = Person(new_person2)
person3 = Person(new_person3)

print(person1)
print(person2)
print(person3)
print(type(person1))