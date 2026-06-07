# Este arquivo diz ao pytest para adicionar a raiz do projeto ao sys.path,
# permitindo imports como `from datasets.apdd import ...`
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
