import torch

from vllm import LLM, SamplingParams

import random

class BasicLLama:
    
    def __init__(self, model_id :str ="meta-llama/Meta-Llama-3.1-70B-Instruct"):
        random.seed(42)
        
        self._model_id = model_id

        print("Device_count: " + str(torch.cuda.device_count()))

        self._generator = LLM(model=self._model_id, tensor_parallel_size=torch.cuda.device_count(), seed=42, max_model_len=7500)

    def batch_generation(self, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], max_new_tokens:int = 256, temperature = 0.6, seed: int = 42):
        # Prepare batch of messages
        batch_messages = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        for system_message, prompt in zip(system_messages, prompts)
        ]

        batch_size = len(system_messages)
        seeds = []

        for i in range(batch_size):
            seeds.append(random.randint(1, 1000000))

        sampling_params_list = [
            SamplingParams(temperature=temperature, top_p=0.9, top_k=50, max_tokens=max_new_tokens, seed=seeds[i])
            for i in range(batch_size)
        ]
        outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False)
        result = [output.outputs[0].text for output in outputs]
        #print(result, flush=True)
        return result