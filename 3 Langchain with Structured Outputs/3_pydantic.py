from typing import Optional
from pydantic import BaseModel

class Person(BaseModel):
    name : str
    age : Optional[int] = None

new_person1 = {"name" : "rahul"}
new_person2 = {"name" : "rahul", "age" : 40}
new_person3 = {"name" : "rahul", "age" : "fourty"}
new_person4 = {"name": "rahul", "age": "40"}

person1 = Person(**new_person1)
person2 = Person(**new_person2)
# person3 = Person(**new_person3) # Pydantic raises a validation error
person4 = Person(**new_person4) # Pydantic coerces the string to an integer

print(person1)
print(person2)
# print(person3)
print(person4)
print(type(person1))