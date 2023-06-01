"""modified from mmcv https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py"""
import ast
import copy
import re
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
import warnings
import inspect
from typing import Any, Dict, Optional
from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module
from pathlib import Path

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


def build_from_cfg(
    cfg: Dict, registry: "Registry", default_args: Optional[Dict] = None
) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be an Registry object, " f"but got {type(registry)}"
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args must be a dict or None, " f"but got {type(default_args)}"
        )

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


def save_config(cfg, cfg_path):
    """Save the running config to a file in the ckpt directory."""
    config = get_config()
    config.save(os.path.join(os.getcwd(), "config.yaml"))
    print("Config saved to ./config.yaml")


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=""):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(v[0]), nargs="+")
        else:
            print(f"cannot parse key {prefix + k} of type {type(v)}")
    return parser


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _pre_substitute_base_vars(filename, temp_config_name):
        """Substitute base variable placehoders to string, so that parsing
        would work."""
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg, base_var_dict, base_cfg):
        """Substitute variable strings to their actual values."""
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file {filename} not found.")
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname
            )
            if platform.system() == "Windows":
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name
            )

            if filename.endswith(".py"):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                # delete imported module
                del sys.modules[temp_module_name]
            else:
                raise TypeError("Only .py type is supported now!")
            # close temp file
            temp_config_file.close()

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = (
                f"The config file {filename} will be deprecated " "in the future."
            )
            if "expected" in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' "instead."
            if "reference" in deprecation_info:
                warning_msg += (
                    " More information can be found at "
                    f'{deprecation_info["reference"]}'
                )
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + "\n"
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )
                base_cfg_dict.update(c)

            # Substitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(
                cfg_dict, base_var_dict, base_cfg_dict
            )

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename, use_predefined_variables=True, import_custom_modules=True):
        if isinstance(filename, Path):
            filename = str(filename)
        cfg_dict, cfg_text = Config._file2dict(filename, use_predefined_variables)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            :obj:`Config`: Config obj.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py')
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)
            # on windows, previous implementation cause error
            # see PR 1077 for details
        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        if isinstance(filename, Path):
            filename = str(filename)

        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename) as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super().__setattr__("_cfg_dict", _cfg_dict)
        super().__setattr__("_filename", _filename)
        super().__setattr__("_text", _text)

    def dump(self, file=None):
        """Dumps config into a file or returns a string representation of the
        config.

        If a file argument is given, saves the config to that file using the
        format defined by the file argument extension.

        Otherwise, returns a string representing the config. The formatting of
        this returned string is defined by the extension of `self.filename`. If
        `self.filename` is not defined, returns a string representation of a
         dict (lowercased and using ' for strings).

        Examples:
            >>> cfg_dict = dict(item1=[1, 2], item2=dict(a=0),
            ...     item3=True, item4='test')
            >>> cfg = Config(cfg_dict=cfg_dict)
            >>> dump_file = "a.py"
            >>> cfg.dump(dump_file)

        Args:
            file (str, optional): Path of the output file where the config
                will be dumped. Defaults to None.
        """
        cfg_dict = super().__getattribute__("_cfg_dict").to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
        elif file.endswith(".py"):
            with open(file, "w", encoding="utf-8") as f:
                f.write(self.pretty_text)

    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            >>> # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__("_cfg_dict")
        super().__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys
            ),
        )


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                    (char == ",")
                    and (pre.count("(") == pre.count(")"))
                    and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # We access the caller using inspect.currentframe() instead of
        # inspect.stack() for performance reasons. See details in PR #1844
        frame = inspect.currentframe()
        # get the frame where `infer_scope()` is called
        infer_scope_caller = frame.f_back.f_back
        filename = inspect.getmodule(infer_scope_caller).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(self, module, module_name=None, force=False):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(
                "module must be a class or a function, " f"but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class or function to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old cli,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or isinstance(name, list)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence"
                f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
