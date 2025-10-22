from setuptools import setup, find_packages

setup(
    name="AnFuelpriceEnv",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch_geometric',
        'pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv',
        'networkx'
        'torchrl',
        'tensordict',
        'pandas',
        'numpy',
        'pyyaml',
        'ipdb',
        'gymnasium',
        'pyyaml',
        'wandb',
        'tensorboard',
        
        
        # ... add any other dependencies here
    ],
    # ... other setup options if needed
)


