from langchain.text_splitter import CharacterTextSplitter

text = """
Quantum computing is a rapidly emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.  Unlike classical bits, which are either 0 or 1, quantum computers use qubits that can exist in a superposition of states, allowing them to process multiple possibilities simultaneously.  This capability, combined with entanglement and interference, enables quantum systems to potentially solve specific tasks, such as integer factorization and drug discovery, exponentially faster than traditional machines.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)

print(result)