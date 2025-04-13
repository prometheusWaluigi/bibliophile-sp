# Contributing to BibliophileSP

Thank you for considering contributing to BibliophileSP! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots

### Suggesting Features

If you have an idea for a new feature, please create an issue with the following information:

- A clear, descriptive title
- A detailed description of the feature
- Why this feature would be useful
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bibliophile-sp.git
   cd bibliophile-sp
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file with your Amazon SP-API credentials.

4. Run tests to ensure everything is working:
   ```bash
   poetry run pytest
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

You can run these tools with:

```bash
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
```

## Testing

Please write tests for any new features or bug fixes. We use pytest for testing.

Run tests with:

```bash
poetry run pytest
```

## Documentation

Please document your code using docstrings and update any relevant documentation files.

## Commit Messages

Write clear, concise commit messages that describe the changes you've made.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

Thank you for your contributions!
