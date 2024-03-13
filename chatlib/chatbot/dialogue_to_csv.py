import csv
from io import StringIO
from numbers import Number
from typing import TypeAlias, Callable

import pendulum

from chatlib.chatbot import DialogueTurn, Dialogue
from chatlib.utils import dict_utils

ColumnValueExtractor: TypeAlias = Callable[[DialogueTurn, int, dict | None], str | Number | None]


class TurnValueExtractor(ColumnValueExtractor):
    def __init__(self, property_name: str | list[str]):
        self.property_name = property_name

    def __call__(self, turn: DialogueTurn, index: int, params: dict | None) -> str | Number | None:
        return dict_utils.get_nested_value(turn.__dict__, self.property_name)


class ParameterValueExtractor(ColumnValueExtractor):
    def __init__(self, key: str | list[str]):
        self.key = key

    def __call__(self, turn: DialogueTurn, index: int, params: dict | None) -> str | Number | None:
        return dict_utils.get_nested_value(params, self.key)


class DialogueCSVWriter:
    def __init__(self, columns: list[str] = None, column_extractors: list[ColumnValueExtractor] | None = None):
        self.columns = ["id", "turn_index", "role", "message", "regenerated"] + (columns or []) + ["timestamp",
                                                                                                   "processing_time"]
        self.column_extractors = [
                                     TurnValueExtractor("id"),
                                     lambda turn, index, params: (index + 1),
                                     lambda turn, index, params: "user" if turn.is_user else "system",
                                     TurnValueExtractor("message"),
                                     lambda turn, index, params: "Yes" if dict_utils.get_nested_value(turn.metadata,
                                                                                                      "regenerated") is True else "No"
                                 ] + (column_extractors or []) + [
                                     lambda turn, index, params: pendulum.from_timestamp(turn.timestamp / 1000,
                                                                                         tz=dict_utils.get_nested_value(
                                                                                             params,
                                                                                             "timezone")).format(
                                         "YYYY-MM-DD hh:mm:ss.SSS zz"),
                                     TurnValueExtractor("processing_time")
                                 ]

    def insertColumn(self, name: str, extractor: ColumnValueExtractor, index: int | None = None) -> 'DialogueCSVWriter':
        if index is not None:
            self.columns.insert(index, name)
            self.column_extractors.insert(index, extractor)
        else:
            self.columns.append(name)
            self.column_extractors.append(extractor)

        return self

    def convert_turn_to_row(self, turn: DialogueTurn, index: int, params: dict | None) -> list[str]:
        return [ext(turn, index, params) for ext in self.column_extractors]

    def _write_csv(self, writer: csv.writer, dialogue: Dialogue, params: dict | None = None):
        writer.writerow(self.columns)
        writer.writerows([self.convert_turn_to_row(turn, i, params) for i, turn in enumerate(dialogue)])

    def _get_csv_writer(self, output) -> csv.writer:
        return csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

    def to_csv_string(self, dialogue: Dialogue, params: dict | None = None) -> str:
        output = StringIO()
        writer = self._get_csv_writer(output)

        self._write_csv(writer, dialogue, params)

        return output.getvalue()

    def to_csv_file(self, path: str, dialogue: Dialogue, params: dict | None = None):
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = self._get_csv_writer(file)
            self._write_csv(writer, dialogue, params)


# Test code
if __name__ == "__main__":
    dialogue = [
        DialogueTurn(
            **{"message": "Hello! How has your day been so far?", "is_user": False, "id": "xZTN4XqofXeMcXml47Um",
               "timestamp": 1691499971615, "processing_time": 1252, "metadata": None}),
        DialogueTurn(**{"message": "regen(", "is_user": True, "id": "8jl7WkqlELsQgcVTSF3G", "timestamp": 1691499977532,
                        "processing_time": None, "metadata": None}),
        DialogueTurn(**{"message": "I'm so sorry to hear that. Can you tell me a bit more about what's troubling you?",
                        "is_user": False, "id": "6pnFJ6960KCnDdRpfzV9", "timestamp": 1691499979388,
                        "processing_time": 1852, "metadata": None})
    ]

    writer = DialogueCSVWriter()

    print(writer.to_csv_string(dialogue))
