from flask import Flask ,request,jsonify
import re
import pandas as pd
from transformers import pipeline
import json
import torch
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering, BertConfig,Trainer, TrainingArguments
import random
import openai
from templates import template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# 

@app.route('/productWithoutModel',methods=['POST'])

def productWithoutModel():
    df = pd.read_csv("./data/fashion_products_csv.csv")
    df.columns = df.columns.str.lower()
    df['price'] = df['price'].astype(str)
    df["rating"] = round(df['rating'],2)
    # query = request.args.get('query')
    request_data = request.get_json()
    query = request_data.get('query')
    if not query:
        return jsonify({"error": "No query provided"})

    query_words = query.split()
    result = df.copy()

    price_range_pattern = r'(\d+(\.\d+)?)\s*(?:to|and)\s*(\d+(\.\d+)?)\s*\w*'
    above_pattern = r'(?:above|more than|above than)\s*(\d+(\.\d+)?)\s*\w*'
    below_pattern = r'(?:below|less than|less|below than|under)\s*(\d+(\.\d+)?)\s*\w*'
    # rating_pattern = r'rating(?:\s*is)?\s*above\s*(\d+(\.\d+)?)'
    rating_pattern = r'rating(?:\s*is)?\s*(less than|greater than|between)?\s*(\d+(\.\d+)?)\s*(?:and)?\s*(\d+(\.\d+)?)?'


    price_range_match = re.search(price_range_pattern, query)
    if price_range_match:
        range_start = float(price_range_match.group(1))
        range_end = float(price_range_match.group(3))
        result = result[result['price'].astype(float).between(range_start, range_end)]
        print(f"Filtering by price range: {range_start} to {range_end}")

    # Check for above price query
    above_match = re.search(above_pattern, query)
    if above_match:
        above_price = float(above_match.group(1))
        result = result[result['price'].astype(float) > above_price]
        print(f"Filtering by price above: {above_price}")

    # Check for below price query
    below_match = re.search(below_pattern, query)
    if below_match:
        below_price = float(below_match.group(1))
        result = result[result['price'].astype(float) < below_price]
        print(f"Filtering by price below: {below_price}")

    rating_match = re.search(rating_pattern, query)
    if rating_match:
        comparison_type = rating_match.group(1)
        if comparison_type == 'less than':
            max_rating = float(rating_match.group(2))
            result = result[result['rating'] < max_rating]
            # Apply filter for ratings less than max_rating
        elif comparison_type == 'greater than':
            min_rating = float(rating_match.group(2))
            result = result[result['rating'] > min_rating]
            # Apply filter for ratings greater than min_rating
        elif comparison_type == 'between':
            min_rating = float(rating_match.group(2))
            max_rating = float(rating_match.group(4))
            result = result[(result['rating'] >= min_rating) & (result['rating'] <= max_rating)]
            # Apply filter for ratings between min_rating and max_rating
        else:
            specific_rating = float(rating_match.group(2))
            result = result[result['rating'] == specific_rating]
            # Apply filter for the specific rating

    for word in query_words:
        # Check if any complete word from the query exists in 'category' column
        if any(result['category'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)):
            print(f"Complete word '{word}' in the query is found in 'category' column.")

            # Filter the data based on 'category' containing the word
            result = result[result['category'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)]
    
    for word in query_words:
        # print("======000000000",query_words)
        # Check if any complete word from the query exists in 'product name' column
        if any(result['product name'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)):
            print(f"Complete word '{word}' in the query is found in 'product name' column.")

            # Filter the data based on 'product name' containing the word
            result = result[result['product name'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)]

        # Check if any complete word from the query exists in 'color' column
        if any(result['color'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)):
            print(f"Complete word '{word}' in the query is found in 'color' column.")

            # Filter the data based on 'color' containing the word
            result = result[result['color'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)]

        # Check if any complete word from the query exists in 'size' column
        if any(result['size'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)):
       
            result = result[result['size'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)]

        # Check if any complete word from the query exists in 'brand' column
        if any(result['brand'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)):

            print(f"Filtering by brand---------")
            q = query_words[query_words.index(word) + 1]
            print(f"Filtering by brand: {q}")
            
            # Filter the data based on 'brand' containing the specified value
            result = result[result['brand'].str.contains(r'\b{}\b'.format(word), case=False, regex=True)]

    return result.to_json(orient='records')

# Example queries, one at a time
query1 = "Show all black Dress of men"
query2 = "Show all Adidas products"
query3 = "Show all products with price less than 50"
query4 = "Show all products with rating above 3"
query5 = "Show all products of size XL"
query6 = "Show all products with price range 40 to 80"
query7 = "Show all products with price above than 50"
query8 = "Show all black Dress of women under 50"
query9 = "show all black dress of women under 50 and their rating is under 4"
# Applying the queries
# print("i hgot===-==================",generate_response(query1))
# print("i hgot===-==================",generate_response(query2))
# print("i hgot===-==================",generate_response(query3))
# print("i hgot===-==================",generate_response(query4))
# print("i hgot===-==================",generate_response(query5))
# print("i hgot===-==================",generate_response(query6))
# print("i hgot===-==================",generate_response(query7))
# print("i hgot===-==================",generate_response(query8))

# @app.route('/http://127.0.0.1:5000/generate_response',methods=['POST'])
@app.route('/productWithTapasModel',methods=['POST'])

def productWithTapasModel():
    request_data = request.get_json()
    query = request_data.get('query')
    data = pd.read_csv('./data/fashion_data_test.csv')

    data = data.astype(str)
    
    if not query:
        return jsonify({"error": "No query provided"})
    
    tapas = pipeline(model="google/tapas-large-finetuned-wtq", tokenizer="google/tapas-large-finetuned-wtq")

    # Use TAPAS to query the dataset and retrieve the coordinates of matching cells
    results = tapas(table=data, query=query)
    matching_rows = [data.iloc[i[0]] for i in results['coordinates']]
    # print("matching_rows",matching_rows)
    
    matching_rows_df = pd.DataFrame(matching_rows) 
    matching_rows_json = matching_rows_df.to_json(orient='records')  # Convert the list to a DataFrame
    
    matching_rows_json_object = json.loads(matching_rows_json)
    print(matching_rows_json_object)  # Check the JSON object
    return matching_rows_json
    
    # return matching_rows.to_json(orient='records')

def evaluate_expression(x, keyword, operator):
    if operator == '<':
        return keyword.lower() in x and x < keyword.lower()
    elif operator == '>':
        return keyword.lower() in x and x > keyword.lower()
    else:
        return keyword.lower() in x
    
@app.route('/productWithBertModel',methods=['POST'])

def productWithBertModel():
    request_data = request.get_json()
    query = request_data.get('query')
    operator = re.findall(r'([<>])', query)
    operator = operator[0] if operator else None   # Retrieve the operator from the JSON input or use '=='
    
    data = pd.read_csv('./data/fashion_products_csv.csv')

    nlp = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # query = "show all black dress"
    # query = "show all black dress with zara brand of women and price is 60 and rating is 2"
    # query1 = "Show all black Dress of women"
    # query = "Show all products of size XL"

    # Split the query into individual keywords
    keywords = query.split()

    # Specify the columns to search in
    columns = ['Brand', 'Product Name', 'Category', 'Color', 'Size','Category',"Price","Rating"]

    # Convert the data to lowercase
    data = data.applymap(lambda x: str(x).lower())

    # Create a list to store the matching keywords
    matching_keywords = []

    # Iterate over the keywords and check if they match any values in the specified columns
    for keyword in keywords:
        if data[columns].applymap(lambda x: keyword.lower() in x).any().any():
            matching_keywords.append(keyword)

    # Print the matching keywords
    print(matching_keywords)

    # Filter the data based on the matching keywords and columns using BERT model
    matching_rows = data
    for keyword in matching_keywords:
        # matching_rows = matching_rows[matching_rows[columns].applymap(lambda x: keyword.lower() in x).any(axis=1)]
        print("--------------operator",operator)
        if operator:
            print("--------------")
            matching_rows = matching_rows[matching_rows[columns].applymap(lambda x: evaluate_expression(x, keyword, operator)).any(axis=1)]
        else:
            print("================")
            matching_rows = matching_rows[matching_rows[columns].applymap(lambda x: keyword.lower() in x).any(axis=1)]

    # Apply BERT sentiment analysis on the filtered rows
    sentiments = []
    for index, row in matching_rows.iterrows():
        text = row['Product Name']  # You can choose a different column for sentiment analysis if needed
        sentiment = nlp(text)[0]
        sentiments.append(sentiment)

    matching_rows['Sentiment'] = sentiments

    # print("matching_rows", matching_rows)

    
    matching_rows_df = pd.DataFrame(matching_rows) 
    matching_rows_json = matching_rows_df.to_json(orient='records')  # Convert the list to a DataFrame
    
    matching_rows_json_object = json.loads(matching_rows_json)
    # print(matching_rows_json_object)  # Check the JSON object
    return matching_rows_json
    
    # return matching_rows.to_json(orient='records')


@app.route('/productWithBertTokenModel',methods=['POST'])

def productWithBertTokenModel():
    request_data = request.get_json()
    search_query = request_data.get('query')
    # Step 1: Prepare the data for BERT
    df = pd.read_csv('./data/fashion_products_csv.csv')
    df['Price'] = df['Price'].astype(str)  # Convert the 'Price' column to string
    df['Rating'] = df['Rating'].astype(str)  
    text_data = df['Product Name'] + " " + df['Brand'] + " " + df['Category'] + " " + df['Color'] + " " + df['Size'] + " " + df['Price']+ " " + df['Rating']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text_data.tolist(), padding=True, truncation=True, return_tensors="pt")

    # Step 2: Load the BERT model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Step 3: Use the BERT model for search and filter mechanism
    outputs = model(**inputs)

    matching_keywords = []

    # search_query = "show all black dress of women"
    # search_query = "dress"
    keywords = search_query.split()  # Change the search query to match Adidas products
    columns = ['Brand', 'Product Name', 'Category', 'Color', 'Size', 'Price', 'Rating']
    # Iterate over the keywords and check if they match any values in the specified columns
    for keyword in keywords:
        if df[columns].applymap(lambda x: keyword.lower() in str(x).lower()).any().any():
            matching_keywords.append(keyword)

    print("matching_keywords", matching_keywords)

    # Step 4: Define filtering criteria based on search query and other attributes
    search_inputs = tokenizer(matching_keywords, padding=True, truncation=True, return_tensors="pt")
    search_outputs = model(**search_inputs)

    # Implement filtering criteria based on the search embeddings and brand name
    threshold = 0.3
    similar_products = []
    search_embedding = torch.mean(search_outputs.last_hidden_state, dim=1).squeeze(0)
    print(f"Search output shape: {search_outputs.last_hidden_state.shape}")

    for i, embedding in enumerate(outputs.last_hidden_state):
        row = df.iloc[i]
        matching_columns = [row[column] for column in columns if isinstance(row[column], str)]
        if all(any(keyword.lower() in str(val).lower() for val in matching_columns) for keyword in matching_keywords):
            similarity = torch.nn.functional.cosine_similarity(search_embedding.unsqueeze(0), torch.mean(embedding, dim=0).unsqueeze(0), dim=1)
            if torch.any(similarity > threshold):
                similar_products.append(row)
    # Display the filtered products
    # for product in similar_products:
    #     print(product)

      
    matching_rows_df = pd.DataFrame(similar_products) 
    matching_rows_json = matching_rows_df.to_json(orient='records')  # Convert the list to a DataFrame
    
    # matching_rows_json_object = json.loads(matching_rows_json)
    # print(matching_rows_json_object)  # Check the JSON object
    return matching_rows_json


@app.route('/getAnswerWithoutModel',methods=['GET'])
def getAnswerWithoutModel():
    questions = [
        "What is your date of birth?",
        "What is your current address?",
        "Are you married? If so, what is your spouse’s full name and date of birth?",
        "Do you have children? If so, what are their names and dates of birth?",
        "Do you have any other dependents or individuals you support financially?",
        "Who do you want to serve as the executor of your will?",
        "List all your assets, such as real estate, bank accounts, investments, and personal property.",
        "Specify how you want your assets distributed among beneficiaries.",
        "Clarify if you have specific bequests, such as sentimental items or charitable donations?",
        "Outline your preferences for your funeral or memorial service."
    ]

    # Generate a meaningful random name
    first_names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Isabella", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Tina", "Ursula", "Victor", "Wendy", "Xander", "Yvonne", "Zachary"]
    random_name = random.choice(first_names)

    def generate_random_name():
        return random.choice(first_names)

    def generate_random_date():
        return f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2020)}"

    def generate_marriage_children_answer():
        is_married = random.choice([True, False])
        if is_married:
            spouse_name = generate_random_name()
            spouse_dob = generate_random_date()
            return f"Yes, {spouse_name}, born on {spouse_dob}"
        else:
            return "No"

    marriage_answer = generate_marriage_children_answer()
    children_answer = generate_marriage_children_answer()

    def generate_random_answers():
        return [
            f"Random Date {random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2020)}",
            f"{random.randint(100, 999)} Random Street, {random_name}, State",
            f"{marriage_answer}",
            f"{children_answer}",
            "None",
            f"{random_name}",
            "Real estate, bank accounts, investments",
            "Equally among all beneficiaries",
            "No specific bequests",
            "Simple funeral service"
        ]

    random_answers = generate_random_answers()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    bert_answers = []
    question_list = []
    random_answers_list = []
    bert_answers_list = []

    for question, answer in zip(questions, random_answers):
        context = answer

        inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        bert_answers.append(bert_answer)
        question_list.append(question)
        random_answers_list.append(answer)

    print("Questions and Answers:")
    # for i, (question, random_answer) in enumerate(zip(questions, random_answers)):
    #     print(f"Question: {question}")
    #     print(f"Random Answer: {random_answer}")
    #     print(f"Bert Answer: {bert_answers[i]}")
    #     print("----------------------------------")

    data = {'Question': question_list, 'Random_Answer': random_answers_list}
    df = pd.DataFrame(data)

    matching_rows_json = df.to_json(orient='records')

    return matching_rows_json

def getAnswer(sample_question):
    # Step 1: Load and preprocess the dataset
    train_data = pd.read_csv('./data/question_dataset.csv')  # Adjust the file path accordingly
    # Step 2: Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')
    # model_save_path = "models/bert_model_trained.pth"
    # model = BertForQuestionAnswering.from_pretrained(model_save_path,local_files_only=True)
    # model.eval()


    # # Select a random context from the training dataset for the predefined question
    sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
    sample_context = random.choice(sample_contexts)

    # Tokenize the sample question and context
    inputs = tokenizer(sample_context, return_tensors='pt', max_length=512, truncation=True)

    # Use the trained model to generate an answer
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    # Return the result
    data = {'Question': [sample_question], 'BERT Answer': [bert_answer]}
    df = pd.DataFrame(data, index=[0])
    return df

@app.route('/getAnswerFromBertModel',methods=['GET'])
def getAnswerFromBertModel():
    
    sample_questions = [
        "What is your date of birth?",
        "What is your current address?",
        "Are you married? If so what is your spouse’s full name and date of birth?",
        "Do you have children? If so what are their names and dates of birth?",
        "Do you have any other dependents or individuals you support financially?",
        "Who do you want to serve as the executor of your will?",
        "List all your assets such as real estate bank accounts investments and personal property.",
        "Specify how you want your assets distributed among beneficiaries",
        "Clarify if you have specific bequests such as sentimental items or charitable donations?",
        "Outline your preferences for your funeral or memorial service"
    ]

    # Step 1: Load and preprocess the dataset
    train_data = pd.read_csv('./data/question_dataset.csv')  # Adjust the file path accordingly

    # Step 2: Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    results = []
    # answers_in_paragraph = []
    for sample_question in sample_questions:
        sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
        if len(sample_contexts) > 0:
            sample_context = random.choice(sample_contexts)
            inputs = tokenizer(sample_question, sample_context, return_tensors='pt', max_length=512, truncation=True)
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            # answers_in_paragraph.append(sample_context)
            # paragraph = " ".join(answers_in_paragraph) 
            data = {'question': [sample_question],'answer': [bert_answer]}
            # print("------------",paragraph)
            # data = {'Question': [sample_question], 'Context': [sample_context], 'BERT Answer': [bert_answer]}
            df = pd.DataFrame(data, index=[0])
            results.append(df)

    final_result = pd.concat(results, ignore_index=True)
    matching_rows_json = final_result.to_json(orient='records')
    # print("===========",paragraph)
    #return matching_rows_json
    # Create a Python dictionary with the "status" and "data" keys and their values
    response_data = {
        "status": "success",  # You can set the status value as needed
        "data": json.loads(matching_rows_json)
    }

    # Serialize the dictionary into a JSON string
    json_response = json.dumps(response_data)

    # Now you have a JSON response with "status" and "data" keys
    print(json_response)
    return json_response

# def getAnswerFromBertModel():
#     # request_data = request.get_json()
#     # sample_question = request_data.get('query')
#     # Step 1: Load and preprocess the dataset
#     train_data = pd.read_csv('./question_dataset.csv')  # Adjust the file path accordingly

#     # Step 2: Initialize the BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#     sample_question = "Who do you want to serve as the executor of your will?"

#     # Select a random context from the training dataset for the predefined question
#     sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
#     sample_context = random.choice(sample_contexts)

#     # Tokenize the sample question and context
#     inputs = tokenizer(sample_question, sample_context, return_tensors='pt', max_length=512, truncation=True)

#     # Use the trained model to generate an answer
#     outputs = model(**inputs)
#     answer_start_scores = outputs.start_logits
#     answer_end_scores = outputs.end_logits

#     answer_start = torch.argmax(answer_start_scores)
#     answer_end = torch.argmax(answer_end_scores) + 1

#     bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

#     # Print the result
#     print(f"Question: {sample_question}")
#     print(f"Context: {sample_context}")
#     print(f"BERT Answer: {bert_answer}")
#     print("----------------------------------")

#     data = {'Question': [sample_question], 'Context': [sample_context], 'BERT Answer': [bert_answer]}
#     df = pd.DataFrame(data, index=[0])

#     matching_rows_json = df.to_json(orient='records')

#     return matching_rows_json


# List of predefined questions


# @app.route('/trainBertModel',methods=['GET'])
# def trainBertModel():
#     # Step 1: Load and preprocess the dataset
#     train_data = pd.read_csv('./question_dataset.csv')  # Adjust the file path accordingly

#     # Step 2: Initialize the BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#     # Step 3: Define the training loop and hyperparameters
#     optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
#     num_epochs = 3  # Adjust the number of epochs as needed

#     # Example training loop
#     for epoch in range(num_epochs):
#         for index, row in train_data.iterrows():
#             question = row['question']
#             context = row['context']

#             inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)

#             outputs = model(**inputs)
#             # Define the loss and backpropagation steps here
#             # Your implementation for training goes here

#         # Print the epoch and any other relevant information
#         print(f"Epoch: {epoch + 1}/{num_epochs}")
        

#     return "Training complete"


# Run the function to get the BERT answer
# result = getAnswerFromBertModel()
# print(result)


@app.route('/getBioDataAnswerFromBertModel',methods=['GET'])
def getBioDataAnswerFromBertModel():
    
    sample_questions = [
        "What state are you in?",
        "What is your full name?",
        "What is your date of birth?",
        "What is your current address?",
        "What is your social security number?",
        "Are you married?",
        "what is your spouse’s full name and date of birth?",
        "Do you have children?",
        "What are their dates of birth of your children?",
        "Who do you want to serve as the executor of your will?",
        "Do you have a secondary choice for executor in case your first choice is unable?",
        "What are the major assets you own?",
        "Do you have bank accounts retirement accounts or other financial accounts?",
        "Do you own any businesses or have interests in partnerships or other entities?",
        "How do you want your assets to be distributed upon your death?",
        "Are there specific bequests you want to leave to certain individuals?",
        "If you have minor children and do you want to set up a trust for their benefit?",
        "If you have minor children and who do you want to serve as their guardian in the event of your death?",
        "Do you have a secondary choice for guardian?",
        "Do you have any outstanding debts such as mortgages loans or credit card balances?",
        "How do you want these debts to be handled upon your death?",
        "Do you have specific wishes for your funeral or memorial service?",
        "Do you have a preferred burial or cremation method?",
        "Have you made any pre-arrangements or pre-payments related to your funeral or burial?",
        "Do you have digital assets that need to be addressed in your will?",
        "How do you want these assets to be handled?",
        "Are there any charitable donations you want to make upon your death?",
        "Do you have any other specific wishes or instructions that should be included in your will?"
    ]

    # Step 1: Load and preprocess the dataset
    train_data = pd.read_csv('./data/updated_question_answers.csv')  # Adjust the file path accordingly

    # Step 2: Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    results = []
    answers_in_paragraph = []
    for sample_question in sample_questions:
        if sample_question == "Are you married?":
            sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
            if len(sample_contexts) > 0:
                sample_context = random.choice(sample_contexts)
                inputs = tokenizer(sample_question, sample_context, return_tensors='pt', max_length=512, truncation=True)
                outputs = model(**inputs)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
                if bert_answer.lower() == "i am not married":  # Check if the answer is "I am not married"
                    data = {'question': [sample_question],'answer': [bert_answer]}
                    df = pd.DataFrame(data, index=[0])
                    results.append(df)
                else:
                    continue  # Skip the subsequent questions
        elif sample_question == "Do you have children?":
            sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
            if len(sample_contexts) > 0:
                sample_context = random.choice(sample_contexts)
                inputs = tokenizer(sample_question, sample_context, return_tensors='pt', max_length=512, truncation=True)
                outputs = model(**inputs)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
                if bert_answer.lower() == "i do not have children":  # Check if the answer is "I do not have children"
                    data = {'question': [sample_question],'answer': [bert_answer]}
                    df = pd.DataFrame(data, index=[0])
                    results.append(df)
                else:
                    continue  # Skip the subsequent question about the birthdates of children
        elif sample_question == "What are their dates of birth of your children?":
            continue  # Skip this question if the person does not have children
        else:
            sample_contexts = train_data[train_data['question'] == sample_question]['context'].values
            if len(sample_contexts) > 0:
                sample_context = random.choice(sample_contexts)
                inputs = tokenizer(sample_question, sample_context, return_tensors='pt', max_length=512, truncation=True)
                outputs = model(**inputs)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                bert_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
                data = {'question': [sample_question],'answer': [bert_answer]}
                df = pd.DataFrame(data, index=[0])
                results.append(df)

    final_result = pd.concat(results, ignore_index=True)
    matching_rows_json = final_result.to_json(orient='records')
    # print("===========",paragraph)
    #return matching_rows_json
    # Create a Python dictionary with the "status" and "data" keys and their values
    response_data = {
        "status": "success",  # You can set the status value as needed
        "data": json.loads(matching_rows_json)
    }

    # Serialize the dictionary into a JSON string
    json_response = json.dumps(response_data)

    # Now you have a JSON response with "status" and "data" keys
    print(json_response)
    return json_response


@app.route('/customWillTemplateOfMarriedPerson',methods=['GET'])
def customWillTemplateOfMarriedPerson():
  

    # Set up your OpenAI API key
    openai.api_key = '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'

    # Define the data
    data = [
        {"answer": "John Smith"},
        {"answer": "123 Main St."},
        {"answer": "Anytown"},
        {"answer": "CA"},
        {"answer": "USA"},
        {"answer": "12345"},
        {"answer": "555-555-5555"},
        {"answer": "January 1, 1970"},
        {"answer": "Software Developer"},
        {"answer": "Married"},
        {"answer": "2 children: Jane Smith (born January 1, 2000) and John Smith Jr. (born January 1, 2005)"},
        {"answer": "Jane Smith"},
        {"answer": "Car (approximate value: $10,000)"},
        {"answer": "John Smith Jr."},
        {"answer": "Boat (approximate value: $20,000)"},
        {"answer": "Jane Doe"},
        {"answer": "Sister-in-law"},
        {"answer": "Jane Smith"},
        {"answer": "I have pre-planned my funeral arrangements."},
        {"answer": "Cremated and ashes scattered"},
        {"answer": "I direct that my organs be donated to the organ donation program."},
        {"answer": "I own a website and social media accounts. I direct my Executor to transfer ownership of these digital assets to my children: Jane Smith and John Smith Jr."},
    ]
   

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a lawyer helping a client draft their last will and testament. Here is the template you want to fill out:\n" + template},
        {"role": "user", "content": f"{data}"}
    ]
    )

    response_answer = response.choices[0].message['content']

    return response_answer
    
@app.route('/customWillTemplateOfPerson',methods=['GET'])
def customWillTemplateOfPerson():
  

    # Set up your OpenAI API key
    openai.api_key = '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'

    # Define the data
    data = [
        {
         
            "answer": "new york"
        },
        {
            "answer": "michael smith"
        },
        {
            "answer": "november 5 1982"
        },
        {
            "answer": "123 maple avenue otherville usa"
        },
        {
            "answer": "234 - 56 - 7890"
        },
        {
            "answer": "i am not married"
        },
        {
            "answer": "lisa brown"
        },
        {
            "answer": "i do not have children"
        },
        {
            "answer": "william williams"
        },
        {
            "answer": "amy johnson my sister"
        },
        {
            "answer": "beachfront property and art collection and antiques"
        },
        {
            "answer": "multiple bank accounts and investments"
        },
        {
            "answer": "board member of a multinational corporation"
        },
        {
            "answer": "among my immediate family members"
        },
        {
            "answer": "i want her to inherit a specific painting that holds sentimental value to our family"
        },
        {
            "answer": "creating a trust for my children ' s well - being"
        },
        {
            "answer": "my sister"
        },
        {
            "answer": "my cousin"
        },
        {
            "answer": "i have"
        },
        {
            "answer": "allocating specific portions of my estate to cover my debts"
        },
        {
            "answer": "i want my funeral"
        },
        {
            "answer": "traditional"
        },
        {
            "answer": "i have a detailed plan in place for my funeral arrangements including pre - arranged services"
        },
        {
            "answer": "i own a significant online collection"
        },
        {
            "answer": "securely transferred to my designated beneficiaries"
        },
        {
            "answer": "my will includes instructions for donating a part of my estate to wildlife conservation and environmental protection organizations"
        },
        {
            "answer": "specific personal possessions"
        }
    ]


    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a lawyer helping a client draft their last will and testament. Here is the template you want to fill out:\n" + template},
        {"role": "user", "content": f"{data}"}
    ]
    )

    response_answer = response.choices[0].message['content']
    return response_answer

    
@app.route('/GPTWillTemplateOfPerson',methods=['GET'])
def GPTWillTemplateOfPerson():
    # Set up your OpenAI API key
    openai.api_key = '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'
    # Define the data
    data = [
        {
         
            "answer": "new york"
        },
        {
            "answer": "michael smith"
        },
        {
            "answer": "november 5 1982"
        },
        {
            "answer": "123 maple avenue otherville usa"
        },
        {
            "answer": "234 - 56 - 7890"
        },
        {
            "answer": "i am not married"
        },
        {
            "answer": "lisa brown"
        },
        {
            "answer": "i do not have children"
        },
        {
            "answer": "william williams"
        },
        {
            "answer": "amy johnson my sister"
        },
        {
            "answer": "beachfront property and art collection and antiques"
        },
        {
            "answer": "multiple bank accounts and investments"
        },
        {
            "answer": "board member of a multinational corporation"
        },
        {
            "answer": "among my immediate family members"
        },
        {
            "answer": "i want her to inherit a specific painting that holds sentimental value to our family"
        },
        {
            "answer": "creating a trust for my children ' s well - being"
        },
        {
            "answer": "my sister"
        },
        {
            "answer": "my cousin"
        },
        {
            "answer": "i have"
        },
        {
            "answer": "allocating specific portions of my estate to cover my debts"
        },
        {
            "answer": "i want my funeral"
        },
        {
            "answer": "traditional"
        },
        {
            "answer": "i have a detailed plan in place for my funeral arrangements including pre - arranged services"
        },
        {
            "answer": "i own a significant online collection"
        },
        {
            "answer": "securely transferred to my designated beneficiaries"
        },
        {
            "answer": "my will includes instructions for donating a part of my estate to wildlife conservation and environmental protection organizations"
        },
        {
            "answer": "specific personal possessions"
        }
    ]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a lawyer helping a client draft their last will and testament.we do not need witness. use my answers and do not give suggestions. Add today date at the end and add my name instead of [Your Name]"},
        {"role": "user", "content": f"{data}"}
    ]
    )
    print(response)
    response_answer = response.choices[0].message['content']
    return response_answer

 
@app.route('/willByChatGpt',methods=['POST'])
def willByChatGpt():
    # Set up your OpenAI API key
    openai.api_key = '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'
    
    #GET QUERY PARAMS 
    request_data = request.get_json()
    data = request_data.get('answer_data')
    
    # response = openai.ChatCompletion.create(
    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a lawyer helping a client draft their last will and testament.we do not need witness. use my answers and do not give suggestions. Add today date at the end and add my name instead of [Your Name]"},
        {"role": "user", "content": f"{data}"}
    ]
    )
    print(response)
    response_answer = response.choices[0].message['content']
    response_data = {
        "status": "success",  # You can set the status value as needed
        "data": response_answer
    }

    # Serialize the dictionary into a JSON string
    json_response = json.dumps(response_data)

    # Now you have a JSON response with "status" and "data" keys
    print(json_response)
    return json_response

if __name__ == '__main__':
    app.run(debug=True)