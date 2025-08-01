# Contributing to CreativeNewEraSlides

Thank you for your interest in contributing to CreativeNewEraSlides! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CreativeNewEra/CreativeNewEraSlides.git
   cd CreativeNewEraSlides
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install .[dev]
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Download AI models**
   ```bash
   # You'll need a Hugging Face token for this
   python setup_models.py
   ```

## Development Workflow

### Code Quality Standards

We maintain high code quality standards using automated tools:

- **Code Formatting**: Black (88 character line length)
- **Linting**: Flake8 with docstring checks
- **Type Checking**: MyPy (ignores missing imports)
- **Security**: Bandit for security analysis
- **Import Sorting**: isort with Black profile

### Running Quality Checks

```bash
# Run all quality checks (same as CI)
black --check . && flake8 . && mypy --ignore-missing-imports . && pytest --maxfail=1 --disable-warnings -q

# Individual tools
black .                              # Format code
blake --check .                      # Check formatting only
flake8 .                            # Lint code
mypy --ignore-missing-imports .     # Type check
bandit -r .                         # Security check
pytest -v                           # Run tests with verbose output
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

```bash
# Install hooks (one time setup)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_main_controller.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Tests are located in the `tests/` directory
- Use PyQt5 mocking to avoid GUI dependencies
- All tests should run headlessly in CI environments
- Follow the existing test patterns in the codebase

## Architecture Guidelines

### MVC Pattern
- **Controllers**: Business logic layer (`controllers/`)
- **UI**: PyQt5 interface definitions (`ui/`)
- **Workers**: Background thread implementations (`workers/`)
- **Utils**: Shared utilities (`utils/`)

### Threading
- Use QThread for CPU/GPU intensive operations
- Keep UI responsive with background processing
- Use PyQt5 signals for component communication

### AI Model Integration
- Follow 16GB VRAM optimization guidelines in CLAUDE.md
- Use the ModelManager singleton for caching
- Implement proper memory management with CUDA cache clearing

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow code quality standards
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   # Run full test suite
   pytest --maxfail=1 --disable-warnings -q
   
   # Run quality checks
   black --check . && flake8 . && mypy --ignore-missing-imports .
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description
   
   - Detailed explanation of changes
   - Why the change was needed
   - Any breaking changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

### PR Requirements

- [ ] All tests pass
- [ ] Code quality checks pass
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] PR description explains the change
- [ ] No merge conflicts

## Coding Standards

### Python Style
- Follow PEP 8 with Black formatting
- Use type hints where beneficial
- Write clear, descriptive docstrings
- Prefer composition over inheritance

### PyQt5 Patterns
- Use signal-slot communication
- Separate UI logic from business logic
- Thread heavy operations with QThread
- Handle UI updates on the main thread only

### AI Model Code
- Follow memory optimization patterns
- Use proper error handling for model operations
- Clear CUDA cache after operations
- Document VRAM requirements

## Issue Reporting

When reporting issues:

1. **Search existing issues** first
2. **Use issue templates** when available
3. **Provide system information**:
   - OS and version
   - Python version
   - PyQt5 version
   - GPU information (if relevant)
4. **Include reproduction steps**
5. **Attach relevant logs** from the application

## Feature Requests

When suggesting features:

1. **Check existing feature requests**
2. **Explain the use case** and why it's needed
3. **Describe the proposed solution**
4. **Consider implementation complexity**
5. **Discuss potential alternatives**

## Getting Help

- **Documentation**: Check CLAUDE.md for project details
- **Issues**: Create an issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions

## License

By contributing to CreativeNewEraSlides, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉