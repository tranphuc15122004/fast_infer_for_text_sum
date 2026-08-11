QUERY_GENERATE = """
You are a great questioner of any text, and are adept at asking valuable and insightful questions. 
Your goal is to generate 1 summary question for the text provided below. 
The generated summary question should try to simulate the tone of human questions as much as possible, 
and make sure that the generated question must be interrogative sentences and a summary question. 
Important! Please make sure this text must be a complete and non-redundant answer to the generated summary question. 
Please directly output the generated summary question, do not output irrelevant text.

DOCUMENT:
{document}
"""


QUERY_PROMPT_QMSUM = """
Refer to the following meeting transcripts and answer the question with brief but complete explanations. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_QMSUM_NORMAL = """
Refer to the following meeting transcripts and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_SQuALITY = """Based on the following story content, provide a comprehensive plot summary that captures the narrative structure and key story elements.

Story content:
{materials}

Question: {question}

Please provide a detailed plot summary that includes:
1. Main characters and their roles/motivations
2. Key events in chronological order
3. Important story developments and turning points
4. Character interactions and relationships
5. Specific details that drive the narrative forward

Write a thorough, well-structured summary that follows the story's progression:"""


QUERY_PROMPT_SQuALITY_NORMAL = """
Refer to the following story and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_GOV = """Based on the following government report content, provide a comprehensive and structured summary that captures the key findings, methodology, and statistical data.

Government report content:
{materials}

Question: {question}

Please provide a detailed summary that includes:
1. Background and context of the study/report
2. Objectives and scope of the investigation
3. Methodology and data sources used
4. Key findings with specific statistics, percentages, and numbers
5. Policy implications and recommendations
6. Time frames and affected populations/jurisdictions

Write a formal, comprehensive summary that maintains the structure and precision of a government report:"""


QUERY_PROMPT_GOV_NORMAL = """
Refer to the following report and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_WCEP = """Based on the following news content, provide a concise summary of the main news event. Focus only on the core facts and avoid unnecessary details.

News content:
{materials}

Question: {question}

Provide a brief, factual summary of the main news event (1-2 sentences):"""


QUERY_PROMPT_WCEP_NORMAL = """
Refer to the following document and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""


QUERY_PROMPT_BOOK = """Please provide a detailed summary based on the following book content:

{materials}

Question: {question}

Provide a comprehensive summary that includes key plot points, character actions, and important details:"""


QUERY_PROMPT_BOOK_NORMAL = """
Refer to the following narrative and answer the question. 

SUPPORTING MATERIALS:
{materials}

QUESTION:
{question}
"""

QUERY_PROMPT = {
    "qmsum": QUERY_PROMPT_QMSUM,
    "wcep": QUERY_PROMPT_WCEP,
    "booksum": QUERY_PROMPT_BOOK,
    "govreport": QUERY_PROMPT_GOV,
    "squality": QUERY_PROMPT_SQuALITY,
    "narrativeqa": QUERY_PROMPT_BOOK_NORMAL
}


QUERY_PROMPT_NORMAL = {
    "qmsum": QUERY_PROMPT_QMSUM_NORMAL,
    "wcep": QUERY_PROMPT_WCEP_NORMAL,
    "booksum": QUERY_PROMPT_BOOK_NORMAL,
    "govreport": QUERY_PROMPT_GOV_NORMAL,
    "squality": QUERY_PROMPT_SQuALITY_NORMAL,
    "narrativeqa": QUERY_PROMPT_BOOK_NORMAL
}


if __name__ == '__main__':
    pass