# Copilot Instructions for XENON fuse

## Project Overview

**fuse** (Framework for Unified Simulation of Events) is a scientific Python package for simulating particle physics events in the XENON dark matter detector. It unifies the XENONnT simulation chain (previously split between epix and WFSim) into a modular, plugin-based architecture built on the [strax framework](https://github.com/AxFoundation/strax).

**Key technologies:**
- Python 3.9+ (package supports 3.9, 3.10, 3.11; CI actively tests 3.10 and 3.11)
- strax/straxen framework for data processing pipelines
- NumPy, Pandas, SciPy for numerical computing
- Numba for performance optimization
- NESTpy for physics yields calculations
- Awkward arrays and uproot for ROOT file handling

## Repository Structure

```
fuse/
├── fuse/                           # Main package
│   ├── __init__.py                 # Package exports and version
│   ├── plugin.py                   # Base plugin classes (FuseBasePlugin, etc.)
│   ├── context.py                  # Simulation context/configuration
│   ├── context_utils.py            # Context helper functions
│   ├── dtypes.py                   # Shared data type definitions
│   ├── common.py                   # Common utility functions
│   ├── vertical_merger_plugin.py   # Vertical merger for data streams
│   └── plugins/                    # Plugin implementations
│       ├── micro_physics/          # Clustering, yields, electric fields
│       ├── detector_physics/       # S1/S2 generation, drift, extraction
│       ├── pmt_and_daq/           # PMT response, DAQ simulation
│       └── truth_information/      # Truth data storage
├── tests/                          # Test suite (pytest-based)
├── docs/                           # Sphinx documentation
├── examples/                       # Jupyter notebooks
├── pyproject.toml                  # Poetry/pip package configuration
├── setup.cfg                       # Flake8, docformatter, doc8 config
└── .pre-commit-config.yaml        # Pre-commit hooks (black, flake8, mypy, etc.)
```

## Development Setup

### Installation

**Standard installation:**
```bash
pip install xenon-fuse
```

**Development installation from source:**
```bash
# SSH (requires SSH key setup):
git clone git@github.com:XENONnT/fuse
# Or HTTPS (easier for first-time users):
git clone https://github.com/XENONnT/fuse.git

cd fuse
pip install -e .  # Editable mode for development
```

**Installing dependencies for testing:**
```bash
pip install pytest coverage coveralls
```

### Required Environment

- Python 3.9, 3.10, or 3.11 (package claims 3.9+ support; CI tests 3.10 and 3.11)
- MongoDB instance for integration tests (via docker or local install)
- utilix configuration file at `~/.xenon_config` (for XENONnT-specific resources)

**Note:** If you see a warning about missing `.xenon_config`, you can ignore it for basic development. Full integration tests require access to XENONnT resources.

## Testing

### Running Tests

**Run all tests:**
```bash
pytest -v
```

**Run with coverage:**
```bash
coverage run --source=fuse -m pytest --durations 0 -v
coverage report
```

**Run specific test file:**
```bash
pytest tests/test_MicroPhysics.py -v
```

**Key test patterns:**
- Tests are **integration tests** that run full plugin chains
- Use `@timeout_decorator.timeout(seconds)` for long-running tests
- Tests use temporary directories for output
- Tests require test data files (downloaded via utilix or included in repo)

### Test Environment Variables

- `TEST_MONGO_URI`: MongoDB connection string (default: `mongodb://localhost:27017/`)
- `FUSE_TEST_SIMULATION_CONFIG`: Simulation config to use (`sr1_dev` or `sr2_dev`)

## Code Style and Linting

### Pre-commit Hooks

The repository uses pre-commit hooks. Install them with:
```bash
pip install pre-commit
pre-commit install
```

**Hooks include:**
- **black** (code formatter): Line length 100, safe mode
- **flake8** (linter): Max line length 100
- **mypy** (type checker): With types-PyYAML, types-tqdm
- **docformatter** (docstring formatter): Google style, 100 char wrap
- **doc8** (documentation linter): For .rst/.md files
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-added-large-files**

### Linting Configuration

**Flake8** (setup.cfg):
- Max line length: 100
- Ignored: E203 (whitespace before ':'), W503 (line break before binary operator)
- Per-file ignores for `__init__.py` files (F401, F403 allowed)

**Black**:
```bash
black --safe --line-length=100 --preview <file>
```

**Flake8**:
```bash
flake8 fuse/
```

**MyPy**:
```bash
mypy fuse/
```

## Plugin Architecture

### Base Classes

All plugins must inherit from one of these:

1. **`FuseBasePlugin`** (most common):
   - Extends `strax.Plugin`
   - Provides deterministic seeding, debug logging
   - Use for standard processing plugins

2. **`FuseBaseDownChunkingPlugin`**:
   - For plugins that need custom chunking behavior
   - Extends `FuseBasePlugin`

3. **`strax.MergeOnlyPlugin`**:
   - For combining multiple data streams without computation
   - Example: `PhotonSummary`, `MicroPhysicsSummary`

### Plugin Structure Template

```python
from fuse.plugin import FuseBasePlugin
import straxen
import strax
import numpy as np

@strax.export  # Required for plugin discovery
class MyPlugin(FuseBasePlugin):
    """Brief description of what this plugin does (single-line for simple cases).
    
    For multi-line docstrings, add a blank line after summary, then provide
    detailed explanation of the physics/processing performed. Follow Google
    style formatting guidelines.
    """
    
    # Required metadata
    __version__ = "0.1.0"  # Semantic versioning (bump when plugin logic changes)
    depends_on = "input_data_name"  # Or tuple for multiple: ("data1", "data2")
    provides = "output_data_name"
    data_kind = "physics_stage"  # Optional: groups related data types
    save_when = strax.SaveWhen.TARGET  # Or EXPLICIT, ALWAYS, NEVER
    
    # Define output data structure
    dtype = [
        (("Description of field [units]", "field_name"), np.float32),
        (("Another field [units]", "another_field"), np.int32),
    ] + strax.time_fields  # Always include time fields
    
    # Configuration options
    some_parameter = straxen.URLConfig(
        default=10.0,
        type=(int, float),
        help="Description of parameter [units]",
    )
    
    resource_parameter = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?fmt=json&take=param_name",
        type=(int, float),
        cache=True,  # Cache resource lookups
        help="Parameter loaded from resource file [units]",
    )
    
    def setup(self):
        """Initialize plugin state."""
        super().setup()  # MUST call this - initializes self.rng for deterministic 
                        # random number generation and sets up debug logging
        # Custom initialization here
        # Access self.rng for random number generation
    
    def compute(self, input_data_name):
        """Main processing logic.
        
        Args:
            input_data_name: Input array from depends_on
            
        Returns:
            Array with dtype matching self.dtype
        """
        # Handle empty input
        if len(input_data_name) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        # Allocate output
        result = np.zeros(len(input_data_name), dtype=self.dtype)
        
        # Process data
        result['field_name'] = self.process(input_data_name)
        result['time'] = input_data_name['time']
        result['endtime'] = input_data_name['endtime']
        
        return result
```

### Plugin Naming Conventions

- **Class names**: PascalCase with action verb + object
  - Examples: `ElectronDrift`, `FindCluster`, `NestYields`, `S1PhotonPropagation`
- **File names**: snake_case matching the class
  - Examples: `electron_drift.py`, `find_cluster.py`
- **Plugin provides**: snake_case descriptive names
  - Examples: `clustered_interactions`, `s1_photons`, `electron_drift_data`
- **Data kinds**: snake_case stage identifiers
  - Examples: `interactions`, `microphysics`, `detector_physics`

### Configuration Pattern

Use `straxen.URLConfig` for all plugin parameters:

```python
# Simple value
max_electrons = straxen.URLConfig(
    default=1000,
    type=int,
    help="Maximum number of electrons to simulate",
)

# From resource file
drift_velocity = straxen.URLConfig(
    default="take://resource://SIMULATION_CONFIG_FILE.json?fmt=json&take=drift_velocity_liquid",
    type=(int, float),
    cache=True,  # Cache the value after first lookup
    help="Drift velocity in liquid xenon [cm/ns]",
)

# Optional parameter
custom_map = straxen.URLConfig(
    default=None,
    type=(str, type(None)),
    help="Path to custom map file, or None to use default",
)

# Non-tracking config (for paths, debug flags)
debug = straxen.URLConfig(
    default=False,
    type=bool,
    track=False,  # Don't track in lineage/provenance
    help="Enable debug logging",
)
```

### Data Type (dtype) Conventions

**Format**: `(("Human-readable description [units]", "field_name"), np.dtype)`

**Rules:**
- Always append `strax.time_fields` (adds `time` and `endtime`)
- Include units in square brackets in descriptions
- Use descriptive field names (snake_case)
- Common types: `np.float32` for physics values, `np.int32` for IDs/counts
- Reuse dtype components from `fuse.dtypes` when possible

**Example:**
```python
from fuse.dtypes import cluster_positions_fields

dtype = [
    (("Number of photons detected", "n_photons"), np.int32),
    (("Total energy deposited [keV]", "energy"), np.float32),
    (("Drift time [ns]", "drift_time"), np.int32),
] + cluster_positions_fields + strax.time_fields
```

### Random Number Generation

**CRITICAL**: Use `self.rng` for all random number generation:

```python
def compute(self, interactions):
    # CORRECT - uses deterministic seed
    random_values = self.rng.uniform(0, 1, size=len(interactions))
    
    # WRONG - not reproducible
    # random_values = np.random.uniform(0, 1, size=len(interactions))
```

The base class automatically initializes `self.rng` with a deterministic seed based on the plugin's lineage and run ID (when `deterministic_seed=True`, which is the default).

## Common Patterns

### Handling Empty Arrays

Always check for empty input:

```python
def compute(self, input_data):
    if len(input_data) == 0:
        return np.zeros(0, dtype=self.dtype)
    # ... process non-empty data
```

### Multiple Dependencies

```python
class MergePlugin(FuseBasePlugin):
    depends_on = ("interactions", "cluster_index")
    
    def compute(self, interactions, cluster_index):
        # Both inputs available
        result = self.process(interactions, cluster_index)
        return result
```

### Numba Optimization

Use `@jit` for performance-critical numerical code:

```python
from numba import jit

class FastPlugin(FuseBasePlugin):
    @staticmethod
    @jit(nopython=True)
    def _fast_computation(array, param):
        """Heavy numerical computation."""
        result = np.zeros(len(array))
        for i in range(len(array)):
            result[i] = array[i] * param
        return result
    
    def compute(self, input_data):
        processed = self._fast_computation(input_data['field'], self.param)
        # ... rest of processing
```

### Stateful Plugins (Cross-Chunk State)

For plugins that need to track state across chunks (e.g., unique IDs):

```python
class ClusteringPlugin(FuseBasePlugin):
    clusters_seen = 1  # Class variable
    
    def compute(self, interactions):
        cluster_ids = self.find_clusters(interactions)
        # Ensure unique IDs across chunks
        cluster_ids += self.clusters_seen
        self.clusters_seen = np.max(cluster_ids) + 1 if len(cluster_ids) > 0 else self.clusters_seen
        # ... rest of processing
```

## Simulation Contexts

The package provides several pre-configured contexts in `fuse.context`:

**Common contexts:**
- `microphysics_context()`: For microphysics simulation (clustering, yields)
- `detector_simulation_context()`: Adds S1/S2 photon propagation
- `xenonnt_fuse_full_chain_simulation()`: Full simulation chain with PMT/DAQ

**Usage:**
```python
import fuse

# Create context with output directory
st = fuse.microphysics_context('./output_data')

# Configure parameters
st.set_config({
    'drift_velocity_liquid': 1.335,  # cm/ns
    'seed_number': 42,
})

# Run simulation
st.make('run_000000', 'clustered_interactions')
```

## Common Issues and Workarounds

### 1. Missing utilix Configuration

**Error:**
```
WARNING - Could not load a configuration file. You can create one at /home/user/.xenon_config
```

**Solution:** For basic development and testing, this warning can be ignored. For full integration tests or using XENONnT-specific resources, create `~/.xenon_config` with appropriate credentials (requires XENONnT access).

### 2. NESTpy Installation (from CI workflow)

NESTpy requires special installation steps due to submodule dependencies:

```bash
# Clone and build nestpy with specific commit (from CI workflow as of Jan 2026)
git clone https://github.com/NESTCollaboration/nestpy.git
cd nestpy
git checkout fb3804e  # Specific commit required for compatibility
git submodule update --init --recursive
cd lib/pybind11
git fetch --tags
git checkout v2.13.0
cd ../../

# Patch CMake version requirement (may need updating for newer CMake versions)
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 2.8.12...3.30)/' CMakeLists.txt

# Install
pip install .
cd ..
rm -rf nestpy
```

**Note:** This is handled automatically in CI but may be needed for local development if you encounter NESTpy issues. The specific commit and CMake version range are from the CI workflow and may need updates over time.

### 3. MongoDB for Testing

Integration tests require a MongoDB instance. Use Docker:

```bash
# Use the same version as CI (MongoDB 4.4.1 is specified in pytest.yml)
# Note: MongoDB 4.4 reached EOL in Feb 2024; consider newer versions for new setups
docker run -d -p 27017:27017 mongo:4.4.1
export TEST_MONGO_URI='mongodb://localhost:27017/'
```

Or use GitHub Actions' MongoDB service (already configured in pytest.yml).

### 4. Strax/Straxen Version Compatibility

The codebase supports multiple strax/straxen versions:

- **Latest dependencies**: Install normally with `pip install .`
- **Minimum versions**: strax >= 1.6.0, straxen >= 2.2.3

The context module has backward compatibility handling:
```python
# Automatic version detection
if hasattr(straxen.contexts, "xnt_common_opts"):
    common_opts = straxen.contexts.xnt_common_opts
else:
    common_opts = straxen.contexts.common_opts  # Older versions
```

### 5. Pre-commit Hook Issues

If pre-commit hooks fail locally:

```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files

# Skip hooks temporarily (not recommended for PRs)
git commit --no-verify
```

### 6. Import Errors with Numba

If you see numba-related import errors, ensure compatible versions:
- numba >= 0.58.1
- numpy compatible with numba (check numba docs for your version; newer numba 0.60+ supports NumPy 2.0+)

Check compatibility: https://numba.readthedocs.io/en/stable/user/installing.html

### 7. Documentation Build

Build documentation locally:

```bash
cd docs
pip install -r doc_requirements.txt
make html
# Output in docs/build/html/
```

## Pull Request Guidelines

When submitting a PR:

1. **Update docstrings** for modified functions/classes
2. **Bump plugin version** (increment `__version__` in modified plugins)
3. **Update documentation** if adding new features
4. **Add tests** for new functionality
5. **Run pre-commit hooks**: `pre-commit run --all-files`
6. **Check if PR resolves an issue**: Reference with "Fixes #123"

**PR template checklist:**
- [ ] Update the docstring(s)
- [ ] Bump plugin version(s)
- [ ] Update the documentation
- [ ] Tests to check the (new) code is working as desired
- [ ] Does it solve one of the GitHub open issues?

**Important:** Put XENONnT-specific information in wiki notes (repo is public).

## Versioning and Releases

- Uses semantic versioning: `MAJOR.MINOR.PATCH`
- Version is stored in:
  - `pyproject.toml` → `version = "X.Y.Z"`
  - `fuse/__init__.py` → `__version__ = "X.Y.Z"`
- Use bumpversion for version updates:
  ```bash
  pip install bumpversion
  bumpversion patch  # or minor, major
  ```
- Release notes maintained in `HISTORY.md`

## Continuous Integration

**GitHub Actions workflows:**

1. **pytest.yml** (main test workflow):
   - Runs on: Ubuntu-latest
   - Python versions: 3.10, 3.11 (package claims 3.9+ support but CI doesn't test 3.9)
   - Dependency versions: latest, lowest (strax 1.6.0, straxen 2.2.3)
   - Tests with coverage reporting to coveralls.io
   - Requires MongoDB service
   - Timeout: 30 minutes per job

2. **python_publish.yml**:
   - Publishes to PyPI on release tags
   - Uses Poetry for build

**Pre-commit.ci:**
- Runs pre-commit hooks on PRs
- Auto-fixes and commits formatting issues
- Weekly auto-updates of hook versions

## Additional Resources

- **Documentation**: https://xenon-fuse.readthedocs.io/
- **strax framework**: https://github.com/AxFoundation/strax
- **straxen**: https://github.com/XENONnT/straxen
- **NESTpy**: https://github.com/NESTCollaboration/nestpy
- **Examples**: See `examples/` directory for Jupyter notebooks

## Key Files Reference

- `fuse/plugin.py`: Base classes for all plugins
- `fuse/context.py`: Pre-configured simulation contexts
- `fuse/dtypes.py`: Reusable data type definitions
- `fuse/common.py`: Utility functions (stable_sort, etc.)
- `pyproject.toml`: Package metadata and dependencies
- `setup.cfg`: Linter configurations
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `.github/workflows/pytest.yml`: CI test configuration
