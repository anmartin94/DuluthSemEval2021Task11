import requests
import openai, os, time


def build_prompt(work_in_progress, related_works):
    prompt = f'My work in progress:\n"{work_in_progress}"\nRetrieved papers:\n'
    for i, paper in enumerate(related_works):
        prompt += f'[{i+1}] {paper["title"]}\n{paper["abstract"]}\n'
    prompt += "Task: Summarize the retrieved papers that are aligned with my work in progress. Please make the summary appropriate for a scholarly article, and write it from the perspective of the author.\n"
    print(len(prompt))
    return prompt

def get_abstract(paper_id):
    paper = requests.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id[:-1]}', params={'fields': 'abstract'})
    paper.raise_for_status()
    results = paper.json()
    print(results)
    return results['abstract']
        
def get_related_works(queries):
    total_recs = []
    for query in queries:
        recommendations = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params={'query': query, 'limit': 3, 'fields': 'title,url,paperId,abstract'})
        recommendations.raise_for_status()
        results = recommendations.json()
        papers = results['data']
        total_recs.extend(papers)
        time.sleep(10)
    return total_recs
    
def get_keyphrases(work_in_progress):
    prompt = f"I want to search a scholarly digital library to find works that are similar to a text. Please provide 5 search queries for the given text. These search queries should be specific enough to find relevant related work, but shouldn't contain more than a few words. Print the queries as a comma-separated list, without newlines, bullet points, or periods. Following is the text:\n\n{work_in_progress}\n"
    
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f'Extract the keyphrases from the following, and reprint them as a comma-separated list:\n\n{response["choices"][0]["message"]["content"]}'}])
    if '\n' in response["choices"][0]["message"]["content"]:
        return response["choices"][0]["message"]["content"].split("\n")
    return response["choices"][0]["message"]["content"].split(", ")
    
def get_summary(text):
    prompt = f"Rewrite the following text so that it is shorter:\n\n{text}"
    return openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])["choices"][0]["message"]["content"]

    
        
def print_papers(query, papers):
    for idx, paper in enumerate(papers):
        print(f"{query} | {paper['paperId']}, {paper['title']}, {paper['url']}")
       
       
API = os.getenv("S2_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

input_data = open("input.txt").readlines()

for work_in_progress in input_data:
    #abstract = get_abstract(work_in_progress)
    #queries = get_keyphrases(abstract)
    print(len(work_in_progress))
    work_in_progress = get_summary(work_in_progress)
    print(len(work_in_progress))
    print(work_in_progress)
    queries = get_keyphrases(work_in_progress)
    print(queries)
    get_related_works(queries)
    related_works = [x for x in get_related_works(queries) if x["abstract"] != 'null']
    prompt = build_prompt(work_in_progress, related_works)
    
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])
    
    print(response["choices"][0]["message"]["content"])

    

    
    

