import json
from itertools import chain
from typing import Dict, Any, Callable, Generator, Optional, TypedDict
import random
import re

import modules.text_generation as text_generation
import modules.shared as shared

class GenerationSettings(TypedDict):
    temperature: float
    max_new_tokens: Optional[int]

# Largely based on and inspired by https://github.com/1rgs/jsonformer
class Jsonformer:
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

    def add_to_progress(self, s: str) -> Generator[str, None, None]:
        self.progress += s
        yield self.progress

    def apply_indent(self) -> Generator[str, None, None]:
        yield from self.add_to_progress(' ' * self.indent)

    def increase_indent(self):
        self.indent += 4

    def decrease_indent(self):
        self.indent -= 4

    def apply_newline(self) -> Generator[str, None, None]:
        yield from self.add_to_progress('\n')

    def apply_key(self, key) -> Generator[str, None, None]:
        yield from self.apply_indent()
        yield from self.add_to_progress(''.join(['"', key, '": ']))

    def generate_object(self, properties: Dict[str, Any]) -> Generator[str, None, None]:
        yield from self.add_to_progress('{')
        properties = list(properties.items())
        if not len(properties):
            yield from self.add_to_progress('}')
            return
        self.increase_indent()
        for i, (key, schema) in enumerate(properties):
            yield from self.apply_newline()
            yield from self.generate_value(schema, key)
            if i != len(properties) - 1:
                yield from self.add_to_progress(',')
        yield from self.apply_newline()
        self.decrease_indent()
        yield from self.apply_indent()
        yield from self.add_to_progress('}')

    def generate_array(self, item_schema: Dict[str, Any]) -> Generator[str, None, None]:
        yield from self.add_to_progress('[')
        yield from self.apply_newline()
        self.increase_indent()

        # Force at least one element in array
        yield from self.apply_indent()
        yield from self.generate_value(item_schema)

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
            will_gen_another_element = ',' in next_tokens[:2]
            if not will_gen_another_element:
                break
            yield from self.add_to_progress(',')
            yield from self.apply_newline()
            yield from self.apply_indent()
            yield from self.generate_value(item_schema)
        yield from self.apply_newline()
        self.decrease_indent()
        yield from self.apply_indent()
        yield from self.add_to_progress(']')
        
    def generate_value(self, schema: Dict[str, Any], key: Optional[str] = None) -> Generator[str, None, None]:
        schema_type = schema["type"]
        if key:
            yield from self.apply_key(key)
        if schema_type == "number":
            yield from self.add_to_progress(str(self.generate_number()))
        elif schema_type == "boolean":
            yield from self.add_to_progress(str(self.generate_boolean()).lower())
        elif schema_type == "string":
            yield from self.add_to_progress('"')
            yield from self.add_to_progress(self.generate_string())
            yield from self.add_to_progress('"')
        elif schema_type == "array":
            # generate array handles its own serialization to self.progress
            yield from self.generate_array(schema["items"])
        elif schema_type == "object":
            # generate_object handles its own serialization to self.progress
            yield from self.generate_object(schema["properties"])
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def get_prompt(self) -> str:
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=self.progress
        )
        return prompt

    def __call__(self) -> Generator[str, None, None]:
        self.progress = ''
        self.indent = 0
        yield from self.generate_object(self.json_schema["properties"])

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

