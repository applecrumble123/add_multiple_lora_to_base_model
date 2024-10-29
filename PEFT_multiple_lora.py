#from huggingface_hub import notebook_login
#from google.colab import userdata
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
from collections import defaultdict
import jsonlines as jl
from peft import LoraConfig, TaskType, PeftModel
#from peft import LoraConfig, TaskType
#from new_peft.src.peft import PeftModel, LoraConfig, TaskType
import torch.nn as nn
import copy

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_token = "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"
#notebook_login()

"""
Enter "huggingface-cli login" in the command prompt
Enter hf_token into the command prompt
Token has not been saved to git credential helper.
Your token has been saved to /home/johnathon/.cache/huggingface/token
"""



# model_name = "google/gemma-2b"

# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# # load the tokenizer based on the model
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# model.to("cuda")
# print(model)

"""
input_text = "What should I do on a trip to Europe?"

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_length=128)
print(tokenizer.decode(outputs[0]))

print("\n========================================\n")

input_text_2 = "Explain the process of photosynthesis in a way a child could understand"
input_ids_2 = tokenizer(input_text_2, return_tensors="pt").to("cuda")
print(input_ids_2)
outputs = model.generate(**input_ids_2, max_length=128)
print(tokenizer.decode(outputs[0]))
"""

dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train[0:1000]")
print("Instruction is {}".format(dataset[0]["instruction"]))
print("Respomse is {}".format(dataset[0]["response"]))
print(dataset)


#### without parameter efficent finetuning
"""
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
"""
"""
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        if example["category"][i] in ['open_qa', 'general_qa']:
            text = "Instruction:\n{}\n\nResponse:\n{}".format(example['instruction'], example['response'])
            output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func = formatting_prompts_func,
)
print("Initialise trainer for training!")

trainer.train()
"""

#### Finetune using parameter efficient finetuning

dataset_2 = load_dataset(dataset_name, split="train")

# get a count of the different categories
categories_count = defaultdict(int)
for _, data in enumerate(dataset_2):
    categories_count[data['category']] += 1
print(categories_count)

# filter out those that do not have any context
filtered_dataset = []
for _, data in enumerate(dataset_2):
    if data["context"]:
        continue
    else:
        text = "Instruction:\n{}\n\nResponse:\n{}".format(data['instruction'], data['response'])
        filtered_dataset.append({"text":text})

print(filtered_dataset[:2])

### uncomment to save the dataset in json file
# with jl.open("dolly-mini-train.jsonl", "w") as writer:
#     writer.write_all(filtered_dataset[0:])

dataset_name = "applecrumble123/databricks-mini"
dataset_custom = load_dataset(dataset_name, split="train[0:1000]")
print(dataset_custom)

### define the parameters
model_name = "google/gemma-2b"
new_model = "gemma-ft"

################################################################
# LoRA parameters
################################################################
# LoRA attention dimension
# also known as rank, higher number for gpu with higher memory space
# lora_r = 64
lora_r = 4

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################
# bitsandbytes parameters
################################################################
# Activate 4-bit precision base model loading
use_4bit = True

# compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################
# Training Arguments Parameters
################################################################

# output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# optimizer to use 
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
# -1 means not using max_steps
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# save checkpoint every x updates steps
save_steps = 25

# log every X updates steps
logging_steps = 25

################################################################
# SFT Parameters
################################################################

# Maximum sequence length to use
# more for higher compute
max_seq_length = 40 # None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = True # False

# Load the entire model on the GPU 0
# device_map = {"": 0}
# "auto" for multple GPUs
device_map = "auto"
#device_map = {"": 0}

# Load QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit, # Activates 4-bit precision loading
    bnb_4bit_quant_type = bnb_4bit_quant_type, # nf4
    bnb_4bit_compute_dtype = compute_dtype, # float16
    bnb_4bit_use_double_quant = use_nested_quant # False
)

# Check GPU compatibility with bfloat116
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("Setting BF16 to True")
        bf16 = True
    else:
        bf16 = False
        
######## Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    quantization_config = bnb_config,
    device_map = device_map
)

#torch.save(model.state_dict(), "/mnt/c/Users/tohji/OneDrive/Desktop/gemma_2b_base.pt")

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)

# end of sequence token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # fix weird overflow issue with fp16 training

######## Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r=lora_r,
    bias = "none",
    task_type="CAUSAL_LM",
    # attention modules
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    fp16 = fp16,
    bf16 = bf16,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_scheduler_type,
    report_to = "tensorboard"
)

print(training_arguments)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset_custom,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    tokenizer = tokenizer,
    args = training_arguments, 
    packing = packing
)


######### uncomment to train the lora model
# train the model
#trainer.train()
# trainer.model.save_pretrained(new_model)
# torch.save(trainer.model.state_dict(), "/mnt/c/Users/tohji/OneDrive/Desktop/lora.pt")


######### uncomment to save only the lora weights
# def save_lora_adapter(trainer, file_path):
#     # Filter LoRA adapter layers
#     lora_adapter = {
#         k: v for k, v in trainer.model.state_dict().items() 
#         if "lora" in k or any(module in k for module in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"])
#     }
#     torch.save(lora_adapter, file_path)
#     print(f"LoRA adapter saved to {file_path}")

# Save only LoRA adapter weights
# save_lora_adapter(trainer, "/mnt/c/Users/tohji/OneDrive/Desktop/lora.pt")


################ Prompt the newly fine-tuned model
# Run inference with the same prompt we used to test the pre-trained model

# same prompt used to test the pretrained model
input_text = "What should I do on a trip to Europe"

# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage = True,
#     return_dict = True,
#     torch_dtype=torch.float16,
#     device_map = device_map
# )

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype=torch.float16,
    #device_map = device_map
)

# uncomment to save the base model
# torch.save(base_model.state_dict(), "/mnt/c/Users/tohji/OneDrive/Desktop/gemma_2b_base.pt")


base_model_weights = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/gemma_2b_base.pt", map_location="cuda")
base_model.load_state_dict(base_model_weights, strict=False)

# Function to check for LoRA weights
def check_for_lora_weights(model):
    # Look for layer names specific to LoRA (commonly "_lora" or similar identifiers)
    lora_layers = [name for name in model.state_dict().keys() if "_lora" in name or "lora" in name.lower()]

    if lora_layers:
        print("LoRA weights found in the model:")
        for name in lora_layers:
            print(f"{name}: {model.state_dict()[name].shape} - Sum of values: {model.state_dict()[name].sum().item()}")
    else:
        print("No LoRA weights found in the model.")


# Check for LoRA weights in the loaded model
check_for_lora_weights(base_model)

# load and merge the LoRA weights with the model weights
# only finetuned the LoRA layers --> need to merge base model and the adapter layers that are finetuned
# merge multple models?? or extend the PEFT class to merge multiple lora
# 2 domains 2 task --> 4 LoRa (finance, healthcare? task specific - QA, train-of-thoughts) --> 4 datasets
# model = PeftModel.from_pretrained(base_model, [new_model, new_model])
# model = model.merge_and_unload()

# lora_model  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/lora.pt")

lora_model_1  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/lora.pt")
lora_model_2  = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/lora2.pt")

def check_for_lora_weights_in_dict(state_dict):
    # Look for layer names specific to LoRA (commonly "_lora" or similar identifiers)
    lora_layers = [name for name in state_dict.keys() if "_lora" in name or "lora" in name.lower()]

    if lora_layers:
        print("LoRA weights found in the model state_dict:")
        for name in lora_layers:
            print(f"{name}: {state_dict[name].shape} - Sum of values: {state_dict[name].sum().item()}")
    else:
        print("No LoRA weights found in the model state_dict.")


lora_models = [lora_model_1, lora_model_2]

#################### add single lora
# # Merge LoRA weights onto the base model
# for name, lora_weight in lora_model_1.items():
#     if name in base_model_weights:
#         # Assume LoRA weights are additive (you may need to adjust if different behavior is needed)
#         base_model_weights[name] += lora_weight
#     else:
#         # If there's a LoRA layer that's not in the base, add it as a new layer
#         base_model_weights[name] = lora_weight
 
 
#################### add multiple lora
for lora_model in lora_models:
    for name, lora_weight in lora_model.items():
        if name in base_model_weights:
            # Merge LoRA weights additively into the base model weights
            base_model_weights[name] += lora_weight
        else:
            # If a LoRA layer is not in the base, add it as a new layer
            base_model_weights[name] = lora_weight 
            
            
# uncomment to save the combined model              
# torch.save(base_model_weights, "/mnt/c/Users/tohji/OneDrive/Desktop/combined_model.pt")

combined_model_path = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/combined_model.pt")
check_for_lora_weights_in_dict(combined_model_path)

# Load the combined weights
combined_model_weights = torch.load("/mnt/c/Users/tohji/OneDrive/Desktop/combined_model.pt", map_location="cuda")

# Load the weights into the model
base_model.load_state_dict(combined_model_weights, strict=False)

# reload the tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
print(input_ids)

outputs = model.generate(**input_ids, max_length=128)
print(tokenizer.decode(outputs[0]))

