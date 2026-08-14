from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate a 5 short questions & answers on the following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into the single document \n notes -> {notes} and \n quiz -> {quiz}",
    input_variables = ["notes", "quiz"]
)

model1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature = 1.2)
model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature = 1.2)
model3 = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature = 1.2)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
        'notes' : prompt1 | model1 | parser,
        'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model3 | parser

chain = parallel_chain | merge_chain

text = """
The provided technical document introduces **AIDR (Agentic Incident Detection & Response)**, an open-weight, multi-layered threat detection system engineered specifically to catch malicious AI agent behaviors by analyzing their full execution transcripts. Traditional endpoint detection and response (EDR) or network monitoring tools only observe downstream system calls—such as file modifications or outbound network requests—making them blind to the underlying intent of an agent. Because an agent’s true motivation and logic reside within its execution transcript (which records the prompt, step-by-step reasoning, tool invocations, and tool outputs), malicious attacks like indirect prompt injections or compromised tool servers often go unnoticed. AIDR addresses this observability gap by reconstructing the causal chain of agent sessions and analyzing them directly.

A central architectural requirement of AIDR is its strict reliance on **locally hosted, open-weight models** rather than third-party hosted APIs. Because agent transcripts frequently contain highly sensitive information—such as database connection strings, cloud access keys, internal source code, and personal customer data—sending this telemetry across a network boundary turn every detection pass into a critical exfiltration risk. To keep sensitive credentials strictly within the local trust boundary while maintaining operational efficiency, the system operates on a dual-tier GPU configuration running open Qwen models via vLLM. A smaller 4-billion parameter model handles low-cost triage, while a larger 30-billion parameter Mixture-of-Experts (MoE) model performs deep reasoning on escalated incidents.

The data pipeline begins with **Dredge**, an out-of-band collector that extracts and normalizes logs from heterogeneous local agent caches (such as JSON lines or SQLite databases) into a standardized event schema. Dredge structures raw logs into three nested data abstractions: individual tool calls (`ToolUsage`), message turns (`ChatMessage`), and full session representations (`AgentEvent`). Crucially, Dredge operates asynchronously from the core execution engine, ensuring that monitoring incurs zero runtime overhead on the agent itself. It correlates asynchronous tool outputs back to their specific call identifiers, producing a unified timeline that mirrors the exact causal sequence of the session.

Threat analysis is divided between two distinct detection tiers to optimize throughput, accuracy, and inference costs. **Sifter (Tier 1)** is a lightweight, high-recall model running `Qwen3-4B` that screens every incoming session. Sifter operates under a strict rule: it can clear benign sessions or escalate suspicious ones, but it never issues convictions. Sessions flagged by Sifter are handed off to **Inspector (Tier 2)**, a `Qwen3-30B-A3B` reasoning agent. Inspector acts as a deep analytical layer, leveraging Model Context Protocol (MCP) servers—including SourceLens (tool source code), ThreatLens (technique retrieval via embeddings), and PolicyLens (enterprise policy checks)—to verify whether the agent’s actions aligned with the user’s original intent or were manipulated by malicious payload injections.

By combining out-of-band telemetry collection, zero-data-leakage open weights, and a two-tiered high-recall to high-precision detection pipeline, AIDR establishes an evidence-grounded defense framework tailored for agentic workflows. It shifts the security boundary from raw system action inspection to explicit intent verification, enabling security teams to catch indirect prompt injections, rogue tool behaviors, and privilege escalation attempts before they manifest as unrecoverable breaches.
"""

result = chain.invoke({"text" : text})

print(result)

chain.get_graph().print_ascii()