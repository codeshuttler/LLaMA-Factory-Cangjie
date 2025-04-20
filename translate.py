from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "saves/Cangjie-Qwen1.5-4B/lora/sft/checkpoint-8000",
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B")

prompt = """
Please convert the java code to CangJie:
```java
// Copyright (c) 2019-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

import java.util. *;
import java.util.stream.*;
import java.lang.*;
public class FIND_INDEX_OF_AN_EXTRA_ELEMENT_PRESENT_IN_ONE_SORTED_ARRAY_1{
static int f_gold ( int arr1 [ ] , int arr2 [ ] , int n ) {
  int index = n ;
  int left = 0 , right = n - 1 ;
  while ( left <= right ) {
    int mid = ( left + right ) / 2 ;
    if ( arr2 [ mid ] == arr1 [ mid ] ) left = mid + 1 ;
    else {
      index = mid ;
      right = mid - 1 ;
    }
  }
  return index ;
}
```
"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！请问有什么可以帮助你的吗？"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
