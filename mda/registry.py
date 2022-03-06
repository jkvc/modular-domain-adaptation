import importlib
import inspect
import os
import sys
from typing import Dict, Type, TypeVar, Union

from omegaconf import OmegaConf


class FromConfigBase:
    def __init__(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}> {self.__dict__!r}"


BaseClassType = TypeVar("BaseClassType")


class Registry:
    def __init__(self, base_class: Type[BaseClassType]):
        self.base_class: Type[BaseClassType] = base_class
        self.registry = {}
        self.class_names = set()

    def register(self, name: str):
        def _register_cls(cls):
            assert name not in self.registry
            assert issubclass(cls, self.base_class)
            assert cls.__name__ not in self.class_names
            self.registry[name] = cls
            self.class_names.add(cls.__name__)

        return _register_cls

    def from_config(
        self, name: str, args: Union[OmegaConf, Dict], **kwargs
    ) -> BaseClassType:
        assert name in self.registry
        cls = self.registry[name]

        if isinstance(args, dict):
            args = OmegaConf.create(args)

        argspecs = inspect.getfullargspec(cls.__init__)
        combined_args = {**args, **kwargs}

        assert argspecs.varargs is None, "*args is not supported"
        required_args = argspecs.args[
            1 : len(argspecs.args) - len(argspecs.defaults or [])
        ]
        for arg_name in required_args:
            assert arg_name in combined_args, f"missing arg_name {arg_name}"

        if argspecs.varkw is None:
            defined_arg_keys = [k for k in argspecs.args if k != "self"]
            combined_args = {
                k: combined_args[k] for k in defined_arg_keys if k in combined_args
            }
        return cls(**combined_args)


def import_all(root: str, base: str):
    for file in os.listdir(root):
        if file.endswith((".py")) and not file.startswith("__"):
            module = file[: file.find(".py")]
            module_name = ".".join([base, module])
            if module_name not in sys.modules:
                importlib.import_module(module_name)
