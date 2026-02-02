# Contributing to SEAL Watermark Experiment

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/seal-watermark-experiment.git
cd seal-watermark-experiment

# Set up development environment
./setup.sh

# Run tests
pytest tests/

# Run quick test
python scripts/run_full_experiment.py --quick-test
```

## 📋 How to Contribute

### Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/yourusername/seal-watermark-experiment/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, GPU, Python version)
   - Error messages/logs

### Suggesting Enhancements

1. Open an issue with the `enhancement` label
2. Clearly describe the proposed feature
3. Explain why it would be useful
4. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Commit with clear messages (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_seal.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions/classes
- Keep functions focused and small
- Use descriptive variable names

```python
# Good
def compute_simhash(embedding: np.ndarray, patch_idx: int) -> int:
    """
    Compute SimHash for given embedding and patch
    
    Args:
        embedding: Semantic embedding vector
        patch_idx: Index of the patch
        
    Returns:
        Hash value as integer
    """
    pass

# Bad
def ch(e, i):
    pass
```

## 📚 Documentation

- Update README.md for new features
- Add docstrings to all public functions
- Include usage examples
- Update CHANGELOG.md

## 🎯 Priority Areas

We especially welcome contributions in:

- **Robustness improvements**: New transform-resistant detection methods
- **Performance**: Speed optimizations for detection
- **Documentation**: Tutorials, examples, better explanations
- **Testing**: More comprehensive test coverage
- **Visualization**: Better result visualization tools

## 💬 Getting Help

- Open an issue with the `question` label
- Join discussions in existing issues
- Contact maintainers: [your-email@nyu.edu]

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Contributors will be acknowledged in:
- README.md
- Release notes
- Academic papers (for significant contributions)

Thank you for helping make SEAL better!
