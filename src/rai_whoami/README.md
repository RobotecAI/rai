# RAI Whoami Configuration Examples

This directory contains examples demonstrating how to use the RAI Whoami configuration loading system.

## Files

- `config.json`: Example configuration file containing robot identity, constitution, and vector database settings
- `load_config.py`: Example script showing how to load configurations from different sources

## Prerequisites

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. For MongoDB example (optional):
   - Install and start MongoDB
   - Set environment variables (optional):
     ```bash
     export MONGODB_USERNAME=your_username
     export MONGODB_PASSWORD=your_password
     ```

## Usage

1. Run the example script:

```bash
python load_config.py
```

This will demonstrate loading configuration from a JSON file. The MongoDB example is commented out by default as it requires a running MongoDB instance.

## Configuration Structure

The example configuration includes:

1. **Robot Identity**

   - Basic information (name, model, serial number)
   - Capabilities list
   - Physical parameters

2. **Robot Constitution**

   - Ethical principles
   - Operational constraints
   - Safety protocols
   - Interaction rules
   - Learning guidelines

3. **Vector Database**
   - Type and connection parameters
   - Collection settings
   - Embedding model configuration

## Customization

You can modify the `config.json` file to match your specific robot's requirements. The configuration validation will ensure all required fields are present and properly formatted.
