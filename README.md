# Auto-ML-Pipeline-Gen

## A Go-based tool for automatically generating end-to-end machine learning pipelines from data schemas.

Auto-ML-Pipeline-Gen is an efficient Go-based command-line interface (CLI) tool designed to streamline the creation of end-to-end machine learning pipelines. By taking a data schema as input, it automatically generates boilerplate code for data ingestion, preprocessing, model training, evaluation, and deployment, significantly reducing development time and ensuring best practices in MLOps.

### ✨ Features

- **Schema-driven Generation**: Generates pipeline components based on user-defined data schemas (e.g., JSON, YAML).
- **Modular Pipeline Components**: Outputs modular and extensible code for each stage of the ML pipeline.
- **Support for Popular ML Frameworks**: Generates code compatible with Python-based ML frameworks like scikit-learn, TensorFlow, and PyTorch.
- **Containerization Ready**: Includes Dockerfile templates for easy containerization and deployment.
- **Configuration Management**: Provides flexible configuration options for pipeline customization.

### 🚀 Getting Started

#### Installation

```bash
go install github.com/Figirs/Auto-ML-Pipeline-Gen
```

#### Usage

```bash
# Generate a new ML pipeline project
auto-ml-pipeline-gen new my_ml_project --schema_file=./data_schema.json

# Example data_schema.json
# {
#   "features": [
#     {"name": "age", "type": "int"},
#     {"name": "salary", "type": "float"},
#     {"name": "city", "type": "string", "encoding": "one-hot"}
#   ],
#   "target": {"name": "churn", "type": "bool"}
# }
```

### 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
