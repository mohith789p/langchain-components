from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Quantum computing is a rapidly emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.  Unlike classical bits, which are either 0 or 1, quantum computers use qubits that can exist in a superposition of states, allowing them to process multiple possibilities simultaneously.  This capability, combined with entanglement and interference, enables quantum systems to potentially solve specific tasks, such as integer factorization and drug discovery, exponentially faster than traditional machines.
"""

splitter_100 = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

splitter_50 = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0,
)

splitter_10 = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
)

chunks_100 = splitter_100.split_text(text)
chunks_50 = splitter_50.split_text(text)
chunks_10 = splitter_10.split_text(text)

print(chunks_100)
print(len(chunks_100))

print(chunks_50)
print(len(chunks_50))

print(chunks_10)
print(len(chunks_10))

# === procedure ===
# first divide the given corpus into paragraph then checks for the chunk_size if bigger breakdown, if smaller combine up and after combining if it becomes bigger then it will not combine and give result.
# next into sentences, thereafter words, thereafter characters