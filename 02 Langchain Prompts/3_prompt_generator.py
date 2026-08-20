from langchain_core.prompts import PromptTemplate

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

template.save("./2 Langchain Prompts/3_template.json")