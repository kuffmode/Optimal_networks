"""Configuration and cluster management for network simulation and optimization."""
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from dask.distributed import Client, LocalCluster, SSHCluster, get_client, default_client

@dataclass
class DaskConfig:
    """Configuration for Dask cluster setup."""
    cluster_type: str = 'local'
    n_workers: Tuple[int, int] = (4, 2)  # (simulation_workers, spo_workers)
    scheduler_options: Dict = None
    worker_options: Dict = None
    slurm_options: Optional[Dict] = None
    dashboard_port: int = 8787  # Default Dask dashboard port

@contextmanager
def dask_cluster(config: DaskConfig):
    """Context manager for setting up and tearing down a Dask cluster."""
    client = None
    try:
        if config.cluster_type == 'local':
            cluster = LocalCluster(
                n_workers=sum(config.n_workers),
                threads_per_worker=1,
                dashboard_address=f':{config.dashboard_port}',  # Enable dashboard
                **(config.scheduler_options or {}),
                **(config.worker_options or {})
            )
            client = Client(cluster)
            dashboard_link = client.dashboard_link
            print(f"\nDask dashboard available at: {dashboard_link}")
            
        elif config.cluster_type == 'slurm':
            if not config.slurm_options:
                raise ValueError("SLURM options required for SLURM cluster")
                
            cluster = SSHCluster(
                scheduler_options={
                    'dashboard_address': f':{config.dashboard_port}',
                    **(config.scheduler_options or {})
                },
                worker_options=config.worker_options,
                **config.slurm_options
            )
            client = Client(cluster)
            print(f"\nDask dashboard available at: {client.dashboard_link}")
            
        else:
            raise ValueError(f"Unknown cluster type: {config.cluster_type}")
            
        yield client
    finally:
        if client:
            client.close()

@contextmanager
def get_or_create_dask_client(config: DaskConfig):
    """Get existing Dask client or create a new one if none exists.
    
    This context manager will:
    1. Try to get an existing client
    2. Create a new one if none exists
    3. Properly clean up only if it created a new client
    """
    client = None
    created_new = False
    
    try:
        # Try to get existing client
        client = get_client()
        print("Using existing Dask client")
    except ValueError:
        # No client exists, create new one
        print("Creating new Dask client")
        with dask_cluster(config) as new_client:
            client = new_client
            created_new = True
            yield client
    else:
        # Found existing client
        yield client
    finally:
        if created_new and client:
            client.close()