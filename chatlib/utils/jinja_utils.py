from jinja2 import Environment, Template, Undefined


def list_with_conjunction(value, conjunction='and'):
    value = [v for v in value if isinstance(v, str)]
    if len(value) == 0:
        return ""
    elif len(value) == 1:
        return value[0]
    elif len(value) == 2:
        return f'{value[0]} {conjunction} {value[1]}'
    else:
        return ', '.join(value[:-1]) + f', {conjunction} {value[-1]}'


__jinja_env = None


def __get_jinja_env() -> Environment:
    global __jinja_env

    if __jinja_env is None:
        __jinja_env = Environment(undefined=Undefined)
        __jinja_env.filters["list_with_conjunction"] = list_with_conjunction
    return __jinja_env


def convert_to_jinja_template(template_string: str) -> Template:
    return __get_jinja_env().from_string(template_string)
