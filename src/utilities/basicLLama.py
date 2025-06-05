import transformers
import torch

from vllm import LLM, SamplingParams

import random

class BasicLLama:
    
    def __init__(self, model_id :str ="meta-llama/Meta-Llama-3.1-70B-Instruct", vllm_inference=False):
        random.seed(42)
        
        self._model_id = model_id

        self._vllm_inference = vllm_inference

        print("Device_count: " + str(torch.cuda.device_count()))

        if self._vllm_inference:
            self._generator = LLM(model=self._model_id, tensor_parallel_size=torch.cuda.device_count(), seed=42, max_model_len=7500)
        else:
            self._generator = transformers.pipeline(
                "text-generation",
                model=self._model_id,
                model_kwargs={"torch_dtype": torch.float16,
                            "attn_implementation": "flash_attention_2",},
                device_map="balanced",
                trust_remote_code=True
            )

            self._terminators = [
                self._generator.tokenizer.eos_token_id,
                self._generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    @model_id.deleter
    def model_id(self):
        del self._model_id


    def basic_generation(self, system_message:str="You are a helpful assistant.", prompt:str="Hi there! What is your name?", max_new_tokens:int = 256, top_k=1, temperature = 0.6, seed=42):
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}]

        
        if temperature == 0:
            do_sample = False
            temperature = None
            top_p = None
        else:
            do_sample = True
            top_p = 0.9


        if self._vllm_inference:
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=50, seed=seed, max_tokens=max_new_tokens)

            outputs = self._generator.chat(messages, sampling_params=sampling_params, use_tqdm=False)
            return outputs.outputs[0].text
        else:

            transformers.set_seed(seed)
            # messages = self._generator.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )

            outputs = self._generator(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=self._terminators,
                pad_token_id=self._generator.tokenizer.eos_token_id,
                do_sample= do_sample,
                temperature= temperature,
                top_p= top_p,
                top_k=50,
            )
            
            return outputs[0]['generated_text'][-1] 
    
    def batch_generation(self, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], max_new_tokens:int = 256, temperature = 0.6, seed: int = 42):
        # Prepare batch of messages
        batch_messages = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        for system_message, prompt in zip(system_messages, prompts)
        ]

        batch_size = len(system_messages)
        seeds = []

        if temperature == 0:
            do_sample = False
            temperature = None
            top_p = None
        else:
            do_sample = True
            top_p = 0.9



        if self._vllm_inference:
            for i in range(batch_size):
                seeds.append(random.randint(1, 1000000))

            sampling_params_list = [
                SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, seed=seeds[i])
                for i in range(batch_size)
            ]
            outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False)
            result = [output.outputs[0].text for output in outputs]
            #print(result, flush=True)
            return result
        else:

            transformers.set_seed(seed)

            outputs = self._generator(
                batch_messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=self._terminators,
                pad_token_id=self._generator.tokenizer.eos_token_id,
                do_sample= do_sample,
                temperature= temperature,
                top_p= top_p,
                top_k=50,
                #temperature = 0.6
                #top_k = top_k
            )
        
            return [output[0]["generated_text"][-1] for output in outputs] 