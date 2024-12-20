from .utils import *
from .prob import *
from .data import *
from .models import *
from .mathlib import *
from .plot import *
from .quantum import *
from .systems import *
from .examples import *

def test_all():
    print_header('mathlib')
    test_mathlib_all()

    print_header('quantum')
    from .quantum import test_quantum_all  # quantum might not be imported yet
    test_quantum_all()