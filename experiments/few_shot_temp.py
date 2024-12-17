
from langchain.llms import Llama3
from langchain.prompts import PromptTemplate
from langchain.chains import FewShotChain



prompt_template = PromptTemplate(
    input_variables=["support", "query"],
    template="Given the following examples:\n\n{support}\n\nClassify the following input:\n\n{query}\n\nThe predicted category is:"
)


support_set = [
    "This is a news article about the economy: 'The stock market is experiencing unprecedented growth.'",
    "A sports article describing a match: 'The team won the championship after a thrilling final.'",
    "An entertainment piece: 'The latest movie received rave reviews from critics.'"
]

query_set = [
    "A news report on climate change.",
    "An exciting basketball match recap.",
    "The latest concert announcement."
]


few_shot_chain = FewShotChain(
    llm=Llama3(),
    prompt=prompt_template,
    support=support_set
)

for query in query_set:
    result = few_shot_chain.run(query=query)
    print(f"Input: '{query}' -> Predicted Category: {result}")