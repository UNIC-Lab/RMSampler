"""Respect caller RNG in the archived non-neural diffusion sampling loop.

Only four literal seed-reset statements are removed from a function copy. The
source file, neural forward methods, network structure, parameters and buffers
are untouched. Both paired experimental controllers use this same correction.
"""
import ast
import inspect
import textwrap
import types


def respect_external_rng(model):
    original = inspect.unwrap(type(model).sample_fn)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    function.decorator_list = []
    names = {'torch.manual_seed','torch.random.manual_seed','np.random.seed','random.seed'}
    removed = []
    retained = []
    for statement in function.body:
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
            call=statement.value
            name=ast.unparse(call.func)
            if name in names:
                assert len(call.args)==1 and isinstance(call.args[0],ast.Constant) and call.args[0].value==0
                assert not call.keywords
                removed.append(name)
                continue
        retained.append(statement)
    assert set(removed)==names and len(removed)==4, 'Unexpected sampler source, do not patch'
    function.body=retained
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    model.sample_fn=types.MethodType(namespace[function.name],model)
    return {'removed_only_literal_rng_resets':removed,
            'neural_forward_or_architecture_changes':0,'pretrained_tensor_changes':0}
