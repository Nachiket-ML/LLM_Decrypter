import os
import pickle
import random
import json 
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

OUTPUT_DIR = './llama_outputs/'
ENCRYPTION_METHODS = ['Caesar', 'Atbash', 'Morse Code', 'Bacon', 'Rail Fence', 'Vigenere', 'Playfair', 'RSA', 'AES']

def parse_arguments():
    """Set up argument parser for the script."""
    parser = argparse.ArgumentParser(description="Process Amazon reviews data for the Two-Tower model.")
    parser.add_argument(
        "--llama_output_dir",
        type=str,
        required=False,
        default='./llama_outputs/',
        help="Path to the llama outputs directory"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        default='meta-llama/Llama-3.2-3B-Instruct',
        help="Model name for the LLM inference requests"
    )
    parser.add_argument(
        '--space_delimeter',
        required=False,
        action='store_true',
        help='Whether to use space delimeter in prompt'
    )
    parser.add_argument(
        '--pass_encryption_method',
        required=False,
        action='store_true',
        help='Whether to pass encryption method into the prompt'
    )
    parser.add_argument(
        '--pass_plaintext',
        required=False,
        action='store_true',
        help='Whether to pass plaintext into the prompt'
    )
    return parser.parse_args()

def fill_prompt_template_encryption_method(ciphertext, encryption_method):
    llama_prompt_template = f"Given the ciphertext: '{ciphertext}' and the encryption method '{encryption_method}', "
    llama_prompt_template += f'decrypt the ciphertext and respond with the plaintext.'
    system_prompt = 'You are a helpful assistant that decrypts ciphertext given the encryption method.'
    return llama_prompt_template, system_prompt

def fill_prompt_template_ciphertext_only(ciphertext, space_delimeter):
    if space_delimeter:
        llama_prompt_template = f"Ciphertext with space delimeters between each character and word: '{ciphertext}'\n"
    else:
        llama_prompt_template = f"Ciphertext: '{ciphertext}'\n"
    llama_prompt_template += "Identify which encryption method from the allowed list was used." 
    system_prompt = 'You are an expert assistant that identifies encryption methods given ciphertext.\n'
    system_prompt += 'Respond with ONLY one of these labels and nothing else:\n'
    system_prompt += 'Caesar, Atbash, Morse Code, Bacon, Rail Fence, Vigenere, Playfair, RSA, AES.'
    return llama_prompt_template, system_prompt

def fill_prompt_template_plaintext(ciphertext, plaintext, space_delimeter):
    if space_delimeter:
        llama_prompt_template = f"Ciphertext with space and comma delimeters between each character and word: '{ciphertext}'\n"
    else:
        llama_prompt_template = f"Ciphertext: '{ciphertext}'\n"
    llama_prompt_template += f"Plaintext: '{plaintext}'\n"
    llama_prompt_template += "Identify which encryption method from the allowed list was used." 
    # llama_prompt_template += "Respond with the encryption method."
    system_prompt = 'You are an expert assistant that identifies encryption methods given ciphertext.\n'
    system_prompt += 'Respond with ONLY one of these labels and nothing else:\n'
    system_prompt += 'Caesar, Atbash, Morse Code, Bacon, Rail Fence, Vigenere, Playfair, RSA, AES.'
    return llama_prompt_template, system_prompt

def get_fname_suffix(pass_encryption_method, pass_plaintext, space_delimeter):
    fname_suffix = ''
    if pass_encryption_method: fname_suffix = 'decrypt_with_encryption_method'
    elif pass_plaintext: fname_suffix = 'get_encryption_method_with_cipher_and_plaintext'
    else: fname_suffix = 'get_encryption_method_with_ciphertext_only'
    if space_delimeter: fname_suffix += '_space_delim_True'
    else:  fname_suffix += '_space_delim_False'
    return fname_suffix

def create_llama_prompts(dataset, output_dir=None, space_delimeter=False, pass_encryption_method=True, pass_plaintext=False):
    fname_suffix = get_fname_suffix(pass_encryption_method, pass_plaintext, space_delimeter)
    if output_dir and os.path.exists(output_dir) and f"llama_prompts_{fname_suffix}.json" in os.listdir(output_dir):
        print(f'Found llama prompts in {output_dir}. Loading from file')
        with open(os.path.join(output_dir, f"llama_prompts_{fname_suffix}.json"), 'r') as f:
            llama_prompts= json.load(f)
            return llama_prompts
    llama_prompts = []
    for i, example in enumerate(dataset):
        if i == 0: print(f"{example['plain_text']}\n{example['cipher_text']}\n{example['algorithm']}")
        ciphertext = example['cipher_text']
        encryption_method = example['algorithm']
        plaintext = example['plain_text']
        if space_delimeter:
            cipherwords = ciphertext.split(' ')
            ciphertext = ', '.join([' '.join(list(word)) for word in cipherwords])
        if pass_encryption_method:
            llama_prompt, system_prompt = fill_prompt_template_encryption_method(ciphertext, encryption_method)
        elif pass_plaintext:
            llama_prompt, system_prompt = fill_prompt_template_plaintext(ciphertext, plaintext, space_delimeter)
        else:
            llama_prompt, system_prompt = fill_prompt_template_ciphertext_only(ciphertext, space_delimeter)
        message = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': llama_prompt}]
        llama_prompts.append(message)
    print(f'First llama prompt: {llama_prompts[0]}')
    if output_dir is None: output_dir = OUTPUT_DIR
    llama_prompts_file= output_dir + f"llama_prompts_{fname_suffix}.json"
    print(f'Writing Llama Prompts to {llama_prompts_file}')
    with open(llama_prompts_file, "w") as outfile:
        json.dump(llama_prompts, outfile)
    return llama_prompts

def create_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # , attn_implementation="flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    generator = pipeline('text-generation', batch_size=24, model=model, tokenizer=tokenizer, use_fast=True, device='cuda')
    print(f'Generator created')
    return generator
    

def flush():
    import torch
    import gc 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_inference_results_generator(generator, llama_prompts, pass_encryption_method):
    print(f'Gathering inference results')
    counts = {method : 0 for method in ENCRYPTION_METHODS}
    responses = ['' for _ in range(len(llama_prompts))]
    dataset = Dataset.from_dict({"prompts": llama_prompts})
    print(f'Creating generator', flush=True)
    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION): 
    if not pass_encryption_method:
        results = generator(KeyDataset(dataset, 'prompts'), batch_size=24, max_new_tokens=3, return_full_text=False)
    else: 
        results = generator(KeyDataset(dataset, 'prompts'), batch_size=24, max_new_tokens=100, return_full_text=False)
    print(f'Llama generator results length: {len(results)}', flush=True)
    assert len(results) == len(llama_prompts)
    prompt_index = 0
    for i, out in enumerate(results): 
        # print(f'Generator output length: {len(out)}', flush=True)
        generated_text = out[0]['generated_text']
        if i % 200 == 0: print(f'Text output: {generated_text}', flush=True)
        # texts = [text for text in batched_text]
        # assert len(batched_text) == 1
        # for j, generated_text in enumerate(batched_text):
        if prompt_index >= len(llama_prompts):
            print(f'Responded to all {len(llama_prompts)} llama prompts.')
            break
        if not pass_encryption_method:
            for method in ENCRYPTION_METHODS:
                if method in generated_text: 
                    responses[prompt_index] = method
                    counts[method] += 1
                    break
            while all(method not in generated_text for method in ENCRYPTION_METHODS):
                print(f'no encryption method found. generated text: {generated_text}, index: {i}, prompt index: {prompt_index}', flush=True)
                prompt = llama_prompts[prompt_index]
                generated_out = generator(prompt, max_new_tokens=3, return_full_text=False)
                generated_text = generated_out[0]['generated_text']
                responses[prompt_index] = generated_text
                for method in ENCRYPTION_METHODS:
                    if method in generated_text: 
                        responses[prompt_index] = method
                        counts[method] += 1
                        break
        else:
            responses[prompt_index] = generated_text.strip()            
        prompt_index += 1
        if i%200 == 0: print(counts)
    print(counts)
    if not pass_encryption_method:
        assert sum(counts.values()) == len(llama_prompts)
    assert len(responses) == len(llama_prompts)
    flush()
    return responses

def get_llama_responses(model_name, llama_prompts, output_dir, space_delimeter, pass_encryption_method, pass_plaintext):
    # llama_responses_dict = {pct: [] for pct in RANDOM_SAMPLING_PCTS}
    # for pct in RANDOM_SAMPLING_PCTS:
    fname_suffix = get_fname_suffix(pass_encryption_method, pass_plaintext, space_delimeter)
    llama_responses_fname = f"llama_responses_{fname_suffix}.json"
    llama_responses_file = output_dir + llama_responses_fname 
    print(f'Llama responses file: {llama_responses_file}')
    print(f"Get llama results for llama prompts length: {len(llama_prompts)}")

    if output_dir and os.path.exists(output_dir) and llama_responses_fname in os.listdir(output_dir):
        print(f'Found llama responses in {output_dir}. Loading from file')
        llama_responses_file = os.path.join(output_dir, llama_responses_fname)
        with open(llama_responses_file, "r") as outfile:
            llama_responses = json.load(outfile)
            return llama_responses
    generator = create_generator(model_name)
    llama_responses = get_inference_results_generator(generator, llama_prompts, pass_encryption_method)
    generator = None
    del generator
    assert len(llama_responses) == len(llama_prompts)
    if output_dir is None: output_dir = OUTPUT_DIR
    print(f'Llama responses output dir: {output_dir}')
    with open(llama_responses_file, "w") as outfile:
        json.dump(llama_responses, outfile)
    return llama_responses

def get_accuracy(llama_responses, dataset):
    correct = 0
    total = len(llama_responses)
    for i, example in enumerate(dataset):
        true_method = example['algorithm']
        if 'Cipher' in true_method:
            cipher_index = true_method.index('Cipher')
            true_method = true_method[:cipher_index].strip()
        predicted_method = llama_responses[i].strip()

        if 'é' in true_method or 'è' in true_method:
            true_method = true_method.replace('é', 'e')
            true_method = true_method.replace('è', 'e')
        if 'é' in predicted_method:
            predicted_method = predicted_method.replace('é', 'e')
            predicted_method = predicted_method.replace('è', 'e')
        if i < 30:
            print(f'Example {i}: True method: {true_method}, Predicted method: {predicted_method}')
        if true_method == predicted_method:
            correct += 1
    accuracy = float(correct/total)
    return accuracy

def main(output_dir, model_name, space_delimeter, pass_encryption_method, pass_plaintext):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # PROMPT LLAMA AND GATHER RESPONSES
    dataset = load_dataset("Sakonii/EncryptionDataset", split="train")
    llama_prompts = create_llama_prompts(dataset, output_dir, space_delimeter, pass_encryption_method, pass_plaintext)
    llama_responses= get_llama_responses(model_name, llama_prompts, output_dir, space_delimeter, pass_encryption_method, pass_plaintext)
    accuracy = get_accuracy(llama_responses, dataset)
    print(f'Accuracy for model name: {model_name}, space delimeter: {space_delimeter}, pass encryption method: {pass_encryption_method}, ' + \
          f'pass plaintext: {pass_plaintext} is {accuracy}')
    # else:
    #     llama_responses = read_llama_responses_dict(output_dir=llama_output_dir)
    # llama_response_item_titles = get_llama_response_item_titles(llama_responses, random_sampling_dict)
    

if __name__ == '__main__':
    args = parse_arguments()
    assert not (args.pass_encryption_method and args.pass_plaintext), "Cannot pass both encryption method and plaintext."
    print(f'Arguments: {args}')
    main(args.llama_output_dir, args.model_name, args.space_delimeter, args.pass_encryption_method, args.pass_plaintext)