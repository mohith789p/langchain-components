from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

code = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"My name is {self.name}, and I am {self.age} years old.")

# Creating an instance
student1 = Student("Amit", 21)
student1.introduce()   
"""

splitter_200 = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 200,
    chunk_overlap = 0
)

splitter_100 = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

chunks_100 = splitter_100.split_text(code)
chunks_200 = splitter_200.split_text(code)

print(chunks_200)
print(len(chunks_200))

print(chunks_100)
print(len(chunks_100))

# === procedure ===
# first divide the given corpus based on class then checks for the chunk_size if bigger breakdown, if smaller combine up and after combining if it becomes bigger then it will not combine and give result.
# next based on def, thereafter sub def, thereafter normal recursive text splitter seperators.