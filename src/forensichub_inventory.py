"""Inspect sibling model declarations without importing torch or model packages.

The inventory includes registered classes, lazy declarations and the model classes
exported to ForensicHub by IMDLBenCo and DeepfakeBench. Abstract classes, wrapper
utilities and post-functions are not experiments.
"""
import ast
import fnmatch
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SCOPE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment_scope.yaml"


def load_experiment_scope(path=None):
    """Read exclusions without importing the model packages."""
    import yaml

    config_path = Path(path or SCOPE_CONFIG)
    scope = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(scope, dict) or not isinstance(scope.get("excluded_models"), list):
        raise ValueError(f"Invalid experiment scope: {config_path}")
    for rule in scope["excluded_models"]:
        if not isinstance(rule, dict) or not rule.get("reason"):
            raise ValueError(f"Every scope exclusion needs a reason: {config_path}")
        for field in ("registry_patterns", "module_patterns", "source_patterns"):
            patterns = rule.get(field, [])
            if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
                raise ValueError(f"Invalid {field} in experiment scope: {config_path}")
    return scope


def exclusion_reason(name, metadata=None, scope=None):
    """Return the configured reason for excluding a name or its source module."""
    metadata = metadata or {}
    scope = scope or load_experiment_scope()
    subjects = {
        "registry_patterns": [name, metadata.get("registry", "")],
        "module_patterns": [metadata.get("import_module", "")],
        "source_patterns": [str(metadata.get("source", "")).replace("\\", "/")],
    }
    for rule in scope["excluded_models"]:
        if any(fnmatch.fnmatchcase(str(subject).lower(), pattern.lower())
               for field, values in subjects.items()
               for subject in values if subject
               for pattern in rule.get(field, [])):
            return rule["reason"]
    return None


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _literal_assignment(tree, variable):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return None


def _exports(package_dir, category):
    path = package_dir / "__init__.py"
    tree = _tree(path)
    names = _literal_assignment(tree, "__all__") or []
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module, alias.name)
    entries = {}
    for name in names:
        if name not in imports:
            continue
        module, class_name = imports[name]
        source = package_dir.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            raise FileNotFoundError(f"Exported model source not found: {source}")
        if not any(isinstance(node, ast.ClassDef) and node.name == class_name
                   for node in _tree(source).body):
            continue  # post-functions are not model experiments
        entries[name] = {"registry": name, "category": category,
                         "source": source, "exported_class": class_name,
                         "declaration": "export"}
    return entries


def get_inventory(workspace_root=None):
    """Return registry-name -> source metadata for all declared model classes."""
    root = Path(workspace_root or WORKSPACE_ROOT)
    package = root / "ForensicHub" / "ForensicHub"
    if not package.is_dir():
        raise FileNotFoundError(f"Sibling ForensicHub checkout not found: {package}")
    entries = {}
    trees = [(package / "common" / "backbones", "backbone")]
    trees += [(package / "tasks" / category / "models", category)
              for category in ("aigc", "document", "deepfake", "fidl")]
    for directory, category in trees:
        for path in sorted(directory.rglob("*.py")):
            tree = _tree(path)
            module = "ForensicHub." + ".".join(path.relative_to(package).with_suffix("").parts)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    function = decorator.func
                    function_name = function.id if isinstance(function, ast.Name) else ""
                    if function_name != "register_model":
                        continue
                    name = ast.literal_eval(decorator.args[0]) if decorator.args else node.name
                    for registry_name in name if isinstance(name, list) else [name]:
                        entries[registry_name] = {
                            "registry": registry_name, "category": category,
                            "source": path, "import_module": module,
                            "exported_class": node.name, "declaration": "decorator",
                        }
            lazy = _literal_assignment(tree, "_lazy_model_map") or {}
            for name, lazy_module in lazy.items():
                entries.setdefault(name, {"registry": name, "category": category,
                                         "source": path, "import_module": lazy_module,
                                         "declaration": "lazy"})
    imdl = root / "IMDLBenCo" / "IMDLBenCo" / "model_zoo"
    deepfake = root / "ForensicHub" / "DeepfakeBench" / "DeepfakeBench" / "training" / "detectors"
    entries.update(_exports(imdl, "imdl"))
    entries.update(_exports(deepfake, "deepfake"))
    return dict(sorted(entries.items()))


def get_scoped_inventory(workspace_root=None, scope=None):
    """Return (experiment declarations, excluded declarations with their reasons).

    get_inventory remains the unfiltered source inventory. Coverage and run
    selection use this filtered view so external declarations do not expand the
    user-defined experiment scope.
    """
    scope = scope or load_experiment_scope()
    included, excluded = {}, {}
    for name, metadata in get_inventory(workspace_root).items():
        reason = exclusion_reason(name, metadata, scope)
        if reason:
            excluded[name] = {**metadata, "excluded_reason": reason}
        else:
            included[name] = metadata
    return included, excluded
