from langchain_core.prompts import PromptTemplate

template= PromptTemplate(
    template="""
        Please summarize the research paper titled "{paper_title}" with the following specifications:
        - Summary Style: {summary_style}
        - Summary Length: {summary_length}
        - Mathematical Details:
            - Include mathematical equations and technical details if the style is "Technical".
            - Avoid complex equations if the style is "Layman".
        - Analogies:
            - Use relatable analogies to explain complex concepts if the style is "Layman".
        If certain information is not available in the paper, please indicate that appropriately.
    """,
    input_variables=["paper_title", "summary_style", "summary_length"],
    validate_template=True
)

template.save("research_paper_summary_template.json")