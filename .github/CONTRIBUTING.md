# Contributing

Thanks for your interest in contributing to this project! This guide will help you get started.

## Bug Reports and Feature Requests

Use the issue templates to report bugs and propose features. Provide as much detail as possible, including steps to reproduce, expected behaviour and screenshots if applicable.

## Development Environment

1. Fork this repository to your own GitHub account and clone the fork locally.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application and ensure it works for you.
4. Run tests (if available) with `pytest` to make sure your environment is set up correctly.

## Submitting Changes

1. Create a new branch off of `main` for your work (`git checkout -b my-feature`).
2. Make your changes following the code style and guidelines below.
3. Commit your changes with clear, descriptive commit messages.
4. Push your branch to your fork and open a pull request against this repository.
5. Ensure all continuous integration checks pass. Address any review comments from maintainers.

## Code Style

We use [Black](https://black.readthedocs.io/en/stable/), [Flake8](https://flake8.pycqa.org/en/latest/) and [Mypy](http://mypy-lang.org/) to enforce code quality. Before submitting a pull request, run:

```bash
black .
flake8 .
mypy .
```

Your changes should not introduce any new linting or type errors.

## Documentation

Please update the README and add docstrings or comments as necessary to explain new functionality. High‑quality documentation makes it easier for others to understand and use your contribution.

## Questions

If you have any questions, feel free to open an issue or start a discussion. We're happy to help!
