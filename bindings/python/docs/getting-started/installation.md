<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

=== "From PyPI (recommended)"

```bash
pip install fastloess
```

=== "From conda-forge"

```bash
conda install -c conda-forge fastloess
```

=== "From Source"

```bash
# Install Rust first: https://rustup.rs/
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/python
pip install maturin
maturin develop --release
```

---

## Verify Installation

```python
import fastloess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

model = fl.Loess()
result = model.fit(x, y)
print("Installed successfully!")
```
