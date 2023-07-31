from jinja2 import Environment, Template

__jinja_env = None


def __get_jinja_env() -> Environment:
    global __jinja_env

    if __jinja_env is None:
        __jinja_env = Environment()
    return __jinja_env


def convert_to_jinja_template(template_string: str) -> Template:
    return __get_jinja_env().from_string(template_string)
