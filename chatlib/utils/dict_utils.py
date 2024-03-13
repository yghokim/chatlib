def set_nested_value(d: dict | None, key: list[str] | str, value) -> dict:
    if d is None:
        d = dict()

    if isinstance(key, list):
        def recur(dictionary: dict, keys: list[str], value):
            if len(keys) == 1:
                dictionary[keys[0]] = value
            else:
                top_key = keys[0]
                if top_key not in dictionary or dictionary[top_key] is None:
                    dictionary[top_key] = dict()
                recur(dictionary[top_key], keys[1:], value)

        recur(d, key, value)
    else:
        d[key] = value

    return d


def get_nested_value(d: dict | None, key: list[str] | str):
    if d is None:
        return None
    elif isinstance(key, list):
        if len(key) == 1:
            return d[key[0]] if key[0] in d else None
        else:
            if key[0] not in d:
                return None
            else:
                return get_nested_value(d[key[0]], key[1:])
    else:
        return d[key] if key in d else None
