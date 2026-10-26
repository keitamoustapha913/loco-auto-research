"""CPU-only complete-domain checks for declared axis-aligned donor geometry.

No candidate import, RNG draws, tensors, image access or protocol modification.
"""
import ast
from pathlib import Path
from benchmark.train_adapter import CandidateContractError


def disjoint_copy_domain(height, width, sides, *, destination_y=None, destination_x=None):
    """All integer origins, or inclusive rectangular subdomains; touching is allowed.

    A donor fits iff one destination margin can contain the donor. The worst
    destination is the nearest-to-centre member on each independent axis. Testing
    that extremum proves the whole rectangular domain, rather than random probes.
    """
    if any(type(v) is not int or v<1 for v in (height,width)):
        raise CandidateContractError('geometry needs positive integer image dimensions')
    if not isinstance(sides,(list,tuple)) or not sides or len(sides)>4096:
        raise CandidateContractError('bounded explicit patch sizes required')
    for side in sides:
        if type(side) is not int or not 1<=side<=min(height,width):
            raise CandidateContractError(f'patch side {side} exceeds image geometry')
        limits=[]
        for length,declared in ((height,destination_y),(width,destination_x)):
            if declared is not None and (not isinstance(declared,(list,tuple)) or len(declared)!=2):
                raise CandidateContractError('destination range must contain inclusive integer minimum and maximum')
            lo,hi=(0,length-side) if declared is None else declared
            if type(lo) is not int or type(hi) is not int or not 0<=lo<=hi<=length-side:
                raise CandidateContractError('destination range outside image')
            limits.append(max(lo,min(hi,(length-side)//2)))
        y,x=limits
        if max(y,height-side-y,x,width-side-x)<side:
            raise CandidateContractError(f'impossible disjoint-copy domain: image={height}x{width}, side={side}, destination=({y},{x}) has no nonoverlapping donor; revise before freezing, never change a frozen recipe')
    return True


def validate_candidate_geometry(source):
    """Literal declarations are checked before import/dispatch.

    Also recognize the historical integer square disjoint_origin primitive; its
    guarded side range and mgrid bound describe its complete declared domain.
    Unknown primitives must provide a declaration and CPU proof, not be guessed.
    """
    tree=ast.parse(Path(source).read_text())
    declarations=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYNTHETIC_EDIT_GEOMETRY' for t in node.targets):
            try:declarations=ast.literal_eval(node.value)
            except (ValueError,TypeError):raise CandidateContractError('SYNTHETIC_EDIT_GEOMETRY must be a literal complete-domain declaration')
    primitive=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='disjoint_origin'),None)
    if declarations is None and primitive is not None:
        ranges=[n for n in ast.walk(primitive) if isinstance(n,ast.Compare) and len(n.ops)==2 and all(isinstance(o,ast.LtE) for o in n.ops) and isinstance(n.left,ast.Constant) and len(n.comparators)==2 and isinstance(n.comparators[0],ast.Name) and n.comparators[0].id=='side' and isinstance(n.comparators[1],ast.Constant)]
        grids=[n for n in ast.walk(primitive) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Attribute) and n.value.attr=='mgrid']
        bounds=[n.upper for grid in grids for n in ast.walk(grid.slice) if isinstance(n,ast.Slice) and n.lower is None and n.step is None and isinstance(n.upper,ast.BinOp) and isinstance(n.upper.op,ast.Sub) and isinstance(n.upper.left,ast.Constant) and type(n.upper.left.value) is int and isinstance(n.upper.right,ast.Name) and n.upper.right.id=='side']
        if len(ranges)==1 and len(bounds)==2 and len({n.left.value for n in bounds})==1:
            r=ranges[0];size=bounds[0].left.value-1
            if type(r.left.value) is not int or type(r.comparators[1].value) is not int or not 0<=r.comparators[1].value-r.left.value<4096:
                raise CandidateContractError('bounded integer side domain required for geometry preflight')
            disjoint_copy_domain(size,size,list(range(r.left.value,r.comparators[1].value+1)))
            if declarations is None:return True
        else:raise CandidateContractError('disjoint donor geometry cannot be verified: declare SYNTHETIC_EDIT_GEOMETRY and exhaustive CPU boundary tests')
    if declarations is not None:
        if not isinstance(declarations,list):raise CandidateContractError('geometry declarations must be a list')
        for d in declarations:
            if not isinstance(d,dict) or not {'kind','height','width','sides'}<=d.keys() or d.keys()-{'kind','height','width','sides','destination_y','destination_x'}:
                raise CandidateContractError('geometry declaration requires kind, height, width, sides and only optional destination ranges')
            if d.get('kind')!='disjoint_square_copy':raise CandidateContractError('unsupported geometry proof; provide an explicit CPU validator')
            disjoint_copy_domain(d['height'],d['width'],d['sides'],destination_y=d.get('destination_y'),destination_x=d.get('destination_x'))
    return True
