import json
from itertools import chain
import queue
import threading
from typing import List, Dict, Any, Callable, Generator, Optional, TypedDict
import random
import re

import modules.text_generation as text_generation
import modules.shared as shared

class GenerationSettings(TypedDict):
    temperature: float
    max_new_tokens: Optional[int]

# Largely based on and inspired by https://github.com/1rgs/jsonformer
class Jsonformer:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        generation_func: Callable[[str, GenerationSettings], Generator[str, None, None]],
        json_schema: Dict[str, Any],
        prompt: str,
        temperature: float,
        max_array_length: int = 10,
    ):
        # Generation func accepts a prompt and generation settings
        self.generation_func = generation_func
        self.json_schema = json_schema
        self.prompt = prompt
        self.temperature = temperature
        self.max_array_length = max_array_length

    def get_next_tokens(
            self, 
            generation_settings: GenerationSettings, 
            stopping_regex: Optional[str] = None, 
            regex_return_group: int = 0,
            prompt_override: Optional[str] = None) -> str:
        prompt = prompt_override or self.get_prompt()
        response_generator = self.generation_func(
            prompt, 
            generation_settings,
        )
        for i, response in enumerate(response_generator):
            print(i)
            print(response)
            if stopping_regex:
                match = re.match(stopping_regex, response)
                if match:
                    return match.group(regex_return_group)
        if stopping_regex:
            raise ValueError("Failed to find match for stopping regex before end of response")
        return response

    def generate_number(self, temperature: Optional[float] = None, iterations=0) -> float:
        settings = {
            'temperature': temperature or self.temperature,
            'max_new_tokens': None,
        }
        stopping_regex = r'(.+)[,\s\]\}]'
        response = self.get_next_tokens(
            settings,
            stopping_regex,
            regex_return_group=1,
        )
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                raise ValueError("Failed to generate a valid number")
            return self.generate_number((temperature or self.temperature) * 1.3, iterations=iterations + 1)

    def generate_boolean(self, temperature: Optional[float] = None, iterations=0) -> bool:
        settings = {
            'temperature': temperature or self.temperature,
            'max_new_tokens': 6,
        }
        # The models have a habit of returning 0/1 for bools sometimes.
        # They usually stop after the first bool to follow their own
        # pattern they've established, but it happens often enough we
        # might as well capture the intent.
        stopping_regex = r'true|false|[01]'
        try:
            response = self.get_next_tokens(settings, stopping_regex)
            if response == 'true' or response == '1': return True
            elif response == 'false' or response == '0': return False
        except ValueError:
            if iterations <= 3: 
                return self.generate_boolean((temperature or self.temperature) * 1.3, iterations=iterations + 1)
            raise
        if iterations <= 3:
            return self.generate_boolean((temperature or self.temperature) * 1.3, iterations = iterations + 1)
        raise ValueError("Failed to generate boolean")


    def generate_string(self, temperature: Optional[float] = None, iterations=0) -> str:
        """ Expects progress to already have a leading `"` populated """
        # This is super inefficient and I should probably figure out a more clever way to
        # tell the model to stop at the end of a string.
        settings = {
            'temperature': temperature or self.temperature,
            'max_new_tokens': None,
        }
        stopping_regex = r'(.*)(?<!\\)"'
        try:
            return self.get_next_tokens(
                settings,
                stopping_regex,
                regex_return_group=1,
            )
        except ValueError:
            if iterations < 3: 
                print("Warning: failed to generate string. Raising temperature...")
                return self.generate_string((temperature or self.temperature * 1.3), iterations = iterations + 1)
            raise

    def add_to_progress(self, s: str):
        self.progress += s
        self.output_queue.put(self.progress)

    def apply_indent(self):
        self.add_to_progress(' ' * self.indent)

    def increase_indent(self):
        self.indent += 4

    def decrease_indent(self):
        self.indent -= 4

    def apply_newline(self):
        self.add_to_progress('\n')

    def apply_key(self, key):
        self.apply_indent()
        self.add_to_progress(''.join(['"', key, '": ']))

    def generate_object(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        obj = {}
        self.add_to_progress('{')
        properties = list(properties.items())
        if not len(properties):
            self.add_to_progress('}')
            return
        self.increase_indent()
        for i, (key, schema) in enumerate(properties):
            self.apply_newline()
            value = self.generate_value(schema, key)
            obj[key] = value
            if i != len(properties) - 1:
                self.add_to_progress(',')
        self.apply_newline()
        self.decrease_indent()
        self.apply_indent()
        self.add_to_progress('}')
        return obj

    def generate_array(self, item_schema: Dict[str, Any]) -> list:
        array = []
        self.add_to_progress('[')
        self.apply_newline()
        self.increase_indent()

        # Force at least one element in array
        self.apply_indent()
        forced_element = self.generate_value(item_schema)
        array.append(forced_element)

        for _ in range(self.max_array_length):
            # Use the model as an oracle as to whether or not it would
            # generate another element by checking whether or not it would
            # next generate a comma.
            # Unfortunately, because models often tokenize end quotes and commas
            # together, if we prompt against a string array that has not been closed,
            # the model will often assume we're at the end of the array. So we have
            # remove the most recent quote marks if present and use that prompt to
            # get the model to accurately tell us what it thinks.
            next_tokens = self.get_next_tokens(
                {
                    'temperature': self.temperature,
                    'max_new_tokens': 3
                },
                prompt_override=self.get_prompt().rstrip('"')
            )
            print(f'NEXT TOKENS: {next_tokens}')
            will_gen_another_element = ',' in next_tokens[:2]
            if not will_gen_another_element:
                break
            self.add_to_progress(',')
            self.apply_newline()
            self.apply_indent()
            new_element = self.generate_value(item_schema)
            array.append(new_element)
        self.apply_newline()
        self.decrease_indent()
        self.apply_indent()
        self.add_to_progress(']')
        return array
        
    def generate_value(self, schema: Dict[str, Any], key: Optional[str] = None) -> Any:
        schema_type = schema["type"]
        if key:
            self.apply_key(key)
        if schema_type == "number":
            num_val = self.generate_number()
            self.add_to_progress(str(num_val))
            return num_val
        elif schema_type == "boolean":
            bool_val = self.generate_boolean()
            self.add_to_progress(str(bool_val).lower())
            return bool_val
        elif schema_type == "string":
            self.add_to_progress('"')
            string_val = self.generate_string()
            self.add_to_progress(string_val)
            self.add_to_progress('"')
            return string_val
        elif schema_type == "array":
            # generate array handles its own serialization to self.progress
            return self.generate_array(schema["items"])
        elif schema_type == "object":
            # generate_object handles its own serialization to self.progress
            return self.generate_object(schema["properties"])
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def get_prompt(self):
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=self.progress
        )
        return prompt

    def _generator(self):
        while True:
            item = self.output_queue.get()
            if item is self.sentinel_object:
                break
            yield item
        print("JSON GENERATION COMPLETE")
        print("GENERATED JSON BELOW")
        print(self.progress)

    def __call__(self) -> Generator[str, None, None]:
        self.progress = ''
        self.output_queue = queue.Queue()
        self.sentinel_object = object()
        self.indent = 0

        threading.Thread(
            target=self.generate_object, 
            args=(self.json_schema["properties"],)
        ).start()
        
        return self._generator()

def custom_generate_reply(question, original_question, seed, state, eos_token, stopping_strings, is_chat=False) -> str:
    """ Overrides the main text generation function """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    # Select text generation function
    generate_func = None
    if shared.model_type in ['rwkv', 'llamacpp']:
        generate_func = text_generation.generate_reply_custom
    elif shared.args.flexgen:
        generate_func = text_generation.generate_reply_flexgen
    else:
        generate_func = text_generation.generate_reply_HF

    # Since we generate many times, we need to lock the seed,
    # so we have to account for when the seed is "random" and
    # lock it for the course of the run. It is still random from
    # the user's perspective, but remains fixed for the repeated
    # generations we run.
    locked_seed = int(seed)
    if locked_seed == -1:
        locked_seed = random.randint(1, 2**31)

    def wrapped_generate_func(wrapped_prompt: str, generation_settings: GenerationSettings):
        state_overrides = {'temperature': generation_settings['temperature']}
        if generation_settings['max_new_tokens'] is not None:
            state_overrides['max_new_tokens'] = generation_settings['max_new_tokens']
        wrapped_state = {
            key: value
            for key, value in chain(state.items(), state_overrides.items())
        }
        return generate_func(wrapped_prompt, original_question, locked_seed, wrapped_state, eos_token, stopping_strings, is_chat)

    jsonformer = Jsonformer(
        generation_func=wrapped_generate_func,
        json_schema=schema,
        prompt=question,
        temperature=state['temperature'],
    )
    
    return jsonformer()

