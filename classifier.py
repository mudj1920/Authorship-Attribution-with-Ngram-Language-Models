#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 01:53:29 2024

@author: mj
"""

import nltk
from nltk.corpus import stopwords
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import KneserNeyInterpolated, Laplace, WittenBellInterpolated
from nltk.util import bigrams,trigrams
from nltk.util import everygrams
from sklearn.model_selection import train_test_split
import string, sys
from sklearn.metrics import classification_report
#from transformers import pipeline
import argparse
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# Function to read file paths
def read_file_paths(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# Function to train a model for a given text file
def train_model(file_path, n=4, full_data=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = [word_tokenize(sent.lower()) for sent in sent_tokenize(text)]
    if full_data:
        train_tokens = tokens
    else:
        train_tokens, test_tokens = train_test_split(tokens, test_size=0.1, random_state=42)
    train_data, vocab = padded_everygram_pipeline(n, train_tokens)
    #model = Laplace(n)
    model = KneserNeyInterpolated(n, discount=0.83)
    #model = WittenBellInterpolated(n)
    model.fit(train_data, vocab)
    return model, tokens if full_data else test_tokens

# Function to prepare test data using everygrams
def prepare_test_data(test_sentences, n):
    return [list(everygrams(list(pad_both_ends(sentence, n)), max_len=n)) for sentence in test_sentences]

# Function to predict the author of a piece of text based on the trained models
def predict_author(test_text, models):
    perplexities = {author: model.perplexity(test_text) for author, model in models.items()}
    return min(perplexities, key=perplexities.get)

# Main
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Authorship Classifier')
    parser.add_argument('authorlist', type=str, help='File with list of author text file paths')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True, help='Choose between generative and discriminative approach')
    parser.add_argument('-test', type=str, help='Test file to predict author', default=None)
    args = parser.parse_args()

    if args.approach == 'discriminative':
        train = load_dataset('Zhongxing0129/authorlist_train')
        test = load_dataset("Zhongxing0129/authorlist_test")
        
        # Initialize a tokenizer for the 'distilbert-base-uncased' model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Define a preprocessing function to tokenize input sentences
        def preprocess_function(examples):
            # Tokenize the sentence(s), truncating them if they exceed the maximum length allowed by the model
            return tokenizer(examples["text"], truncation=True)
        
        # Apply the preprocess_function to the entire dataset, processing in batches for efficiency
        tokenized_train = train.map(preprocess_function, batched=True)
        tokenized_test = test.map(preprocess_function, batched=True)
        
        # Initialize a data collator that will dynamically pad the batched samples to the maximum length in each batch
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Load the accuracy metric from the 'evaluate' library to evaluate model performance
        accuracy = evaluate.load("accuracy")

        def compute_label_wise_accuracy(predictions, references, label_indices):
            label_wise_accuracy = {}
            for label, index in label_indices.items():
                # Select only the predictions and references for the current label
                relevant_predictions = predictions[references == index]
                relevant_references = references[references == index]
        
                # Calculate accuracy for the current label
                label_acc = accuracy_score(relevant_references, relevant_predictions)
                label_wise_accuracy[label] = label_acc
            return label_wise_accuracy

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            overall_accuracy = accuracy_score(labels, predictions)
    
            label_wise_accuracy = compute_label_wise_accuracy(predictions, labels, label2id)
    
            return {"overall_accuracy": overall_accuracy, **label_wise_accuracy}
        
        labels = ['Austen', 'Wilde','Tolstoy','Dickens']
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in id2label.items()}
        
        # Initialize a model for sequence classification based on 'distilbert-base-uncased',
        # specifying the number of labels and the label mappings
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id)
        
        # Define training arguments for the Trainer
        training_args = TrainingArguments(
            output_dir='hw2',
            learning_rate=2e-5,
            per_device_train_batch_size=20,
            per_device_eval_batch_size=20,
            num_train_epochs=4,
            weight_decay=0.02,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        )
    
        #Instantiating the Trainer class
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train['train'],
            eval_dataset=tokenized_test['train'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
        #Training the model
        trainer.train()

        # Initialize counters for tracking mismatches and indexing
        count = 0
        i = 0

        # Load tokenizer and model from the specified model name
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = model

        # Ensure you define 'device' based on the availability of CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move the model to the appropriate device

        validation = tokenized_test['train']

        # Loop until 10 mismatches between predicted and actual labels are found
        while count != 5:
            # Extract the sentence to be classified
            text = validation['text'][i]

            # Tokenize the text and prepare it for the model
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the same device as the model

            # Perform inference without computing gradients for efficiency
            with torch.no_grad():
                logits = model(**inputs).logits

            # Determine the predicted class ID based on the logits
            predicted_class_id = logits.argmax().item()

            # Check if the prediction matches the actual label
            if predicted_class_id != validation['label'][i]:
                # Print the sentence and confidence scores for incorrect predictions
                print(text)
                print('Confidence score:', torch.nn.functional.softmax(logits, dim=1))
                print('Predicted:', model.config.id2label[predicted_class_id], "The actual is:", model.config.id2label[validation['label'][i]])

                # Increment the mismatch counter
                count += 1

            # Move to the next sentence
            i += 1
            
    else:  # args.approach == 'generative'
        authors = ['austen', 'dickens', 'tolstoy', 'wilde']
        file_paths = read_file_paths(args.authorlist + '.txt')
        models = {}
        test_data_by_author = {}

        if args.test:
            
            print('training LMs... (this may take a while)')
            # Train models on the full dataset
            for author, path in zip(authors, file_paths):
                model, _ = train_model(path, full_data=True)
                models[author] = model
            # Predict author for the test file
            with open(args.test+'.txt', 'r', encoding='utf-8') as file:
                test_text = file.read()
            test_tokens = [list(word_tokenize(sent.lower())) for sent in sent_tokenize(test_text)]
            prepared_test_data = prepare_test_data(test_tokens, n=4)
            for test_text in prepared_test_data:
                predicted_author = predict_author(test_text, models)
                print(f'{predicted_author}')
        else:
            print('splitting into training and development...')
            print('training LMs... (this may take a while)')
            print('Results on dev set:')
            # Train models and evaluate accuracy as per existing code
            for author, path in zip(authors, file_paths):
                model, test_data = train_model(path)
                models[author] = model
                test_data_by_author[author] = test_data

            # Evaluate each model on the test data of each author and calculate accuracy
            author_perplexities = {}
            for actual_author, test_sentences in test_data_by_author.items():
                correct_predictions = 0
                total_predictions = 0
                total_perplexity = 0
                failure_cases = []
                prepared_test_data = prepare_test_data(test_sentences, n=4)
                for idx, test_text in enumerate(prepared_test_data):
                    perplexity = models[actual_author].perplexity(test_text)
                    total_perplexity += perplexity
                    predicted_author = predict_author(test_text, models)
                    if predicted_author == actual_author:
                        correct_predictions += 1
                    else:
                        if(len(failure_cases)<2):
                            original_sentence = ' '.join(test_sentences[idx])
                            failure_cases.append(original_sentence)
                    total_predictions += 1
                accuracy = (correct_predictions / total_predictions) * 100  # Convert to percentage
                average_perplexity = total_perplexity / len(prepared_test_data)
                # Store the average perplexity for each author
                author_perplexities[actual_author] = average_perplexity
                print(f'{actual_author.lower():<10}{accuracy:>5.1f}% correct')
                #print(f'Failure cases for {actual_author}:')
                #for case in failure_cases:
                    #print(case)
                    
            prompts = ['small', 'correct', 'blood', 'here', 'together']
            for i in prompts:
                generated_text_austen = models['wilde'].generate(10, text_seed = i, random_seed = 9)
                #print("Text generated by wilde:", generated_text_austen)
                #print("Perplexity for austen:",models['austen'].perplexity(list(everygrams(list(pad_both_ends(generated_text_austen, 4)), max_len=4))))
                #print("Perplexity for dickens:",models['dickens'].perplexity(list(everygrams(list(pad_both_ends(generated_text_austen, 4)), max_len=4))))
                #print("Perplexity for tolstoy:",models['tolstoy'].perplexity(list(everygrams(list(pad_both_ends(generated_text_austen, 4)), max_len=4))))
                #print("Perplexity for wilde:",models['wilde'].perplexity(list(everygrams(list(pad_both_ends(generated_text_austen, 4)), max_len=4))))
            #for author, perplexity in author_perplexities.items():
                #print(f'Perplexity for {author}: {perplexity:.2f}')
                
