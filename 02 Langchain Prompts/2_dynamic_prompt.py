from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    model_kwargs={
        "cache_dir" : "./models"
    },
    pipeline_kwargs={
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template =
    """
    Act as a principal researcher and academic editor specializing in high-impact publishing. Your objective is to draft a rigorous, publication-ready research paper based on the provided topic and constraints.
    ---

    ### Research Context
    - **Paper Title / Topic:** {paper_title}
    - **Style / Methodology / Target Tone:** {style_tone}
    - **Target Length / Scope:** {length}  
    ---

    ### Execution Instructions
    1. **Structural & Content Architecture**
    - **Context & Problem Statement:** Clearly define the core domain, identify the primary research gap, and state the core objective based on the title.
    - **Methodological Alignment:** Ensure argument structure, terminology, and analysis strictly conform to the specified style, methodology, and target tone.
    - **Technical Rigor:** Incorporate formal domain language, precise operational definitions, logical transitions, and structured argumentation. Avoid superficial hand-waving or conversational filler.
    - **Scope Control:** Maintain systematic progression across sections (Abstract, Introduction, Methodology/Framework, Discussion) proportional to the requested target length.

    2. **Style & Constraints**
    - Maintain a formal academic register throughout.
    - Do not include meta-commentary, introductory filler, or conversational intros. Begin directly with the draft paper.
    ---

    ### Output Requirements
    Deliver a continuous, high-density academic draft that fully addresses the paper title, aligns with the specified methodology/tone, and meets the target length requirement.
    """,
    input_variables = ["paper_title", "style_tone", "length"]
)

st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt = template.invoke({
    "paper_title" : paper_input,
    "style_tone" : style_input,
    "length" : length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)