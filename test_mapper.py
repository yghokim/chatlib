import asyncio
import json
from time import perf_counter

from chatlib.tool.converter import dict_to_json_str_converter, json_str_to_dict_converter

from chatlib.chatbot import ChatCompletionParams
from chatlib.llm.integration import GPTChatCompletionAPI, ChatGPTModel
from chatlib.tool.versatile_mapper import ChatCompletionFewShotMapper, ChatCompletionFewShotMapperParams, \
    MapperInputOutputPair

if __name__ == "__main__":


    mapper = ChatCompletionFewShotMapper[str, dict, ChatCompletionFewShotMapperParams](
        api=GPTChatCompletionAPI(),
        instruction_generator="Given the following plain text paragraph, extract the information into a JSON object that includes the name, age, and list of hobbies. Note that the hobbies should contain what the person is currently enjoying.",
        input_str_converter=None,
        output_str_converter=dict_to_json_str_converter,
        str_output_converter=json_str_to_dict_converter
    )

    params = ChatCompletionFewShotMapperParams(model=ChatGPTModel.GPT_3_5_0613, api_params=ChatCompletionParams())

    async def run():
        mapper.api.authorize()

        start_ts = perf_counter()

        result = await mapper.run(examples=[
            MapperInputOutputPair(
                input="John Doe is 29 years old and enjoys reading, cycling, and hiking. He recently picked up photography, adding it to his list of hobbies.",
                output={
                    "name": "John Doe",
                    "age": 29,
                    "hobbies": ["reading", "cycling", "hiking", "photography"]
                }
            )
        ], input="Emily Johnson is 25 years old and has a variety of interests, including painting, running, baking, and playing the guitar. However, recently she stopped running because she broke her leg.",
        params=params
        )

        end_ts = perf_counter()

        print(f"Elapsed time: {int((end_ts - start_ts) * 1000)} millis.")
        print(result)



    asyncio.run(run())