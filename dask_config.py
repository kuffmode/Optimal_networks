"""Configuration and cluster management for network simulation and optimization."""
from typing import Optional
from dataclasses import dataclass
from dask.distributed import Client, LocalCluster, get_client

# Global Dask client instance
GLOBAL_DASK_CLIENT: Optional[Client] = None

@dataclass
class DaskConfig:
    """Configuration for Dask cluster setup."""
    cluster_type: str = 'local'
    n_workers: tuple = (16, 16)  # (simulation_workers, optimization_workers)
    dashboard_port: int = 8787  # Default Dask dashboard port

def get_or_create_dask_client(config: DaskConfig) -> Client:
    """Ensure a single Dask client is used globally, recreating if closed."""
    global GLOBAL_DASK_CLIENT
    # Check if existing client is still operational
    if GLOBAL_DASK_CLIENT is not None:
        try:
            # Simple check to see if the client is responsive
            GLOBAL_DASK_CLIENT.get_versions(check=True)
            return GLOBAL_DASK_CLIENT
        except Exception:
            # Client is closed or in a bad state, reset it
            GLOBAL_DASK_CLIENT = None

    if config.cluster_type == 'local':
        cluster = LocalCluster(
            n_workers=sum(config.n_workers),
            threads_per_worker=1,
            dashboard_address=f":{config.dashboard_port}",
        )
        GLOBAL_DASK_CLIENT = Client(cluster)
    elif config.cluster_type == 'slurm':
        raise NotImplementedError("SLURM support needs additional setup.")

    print(f"Dask dashboard available at: {GLOBAL_DASK_CLIENT.dashboard_link}")
    return GLOBAL_DASK_CLIENT

def close_dask_client():
    """Close the global Dask client if it exists."""
    global GLOBAL_DASK_CLIENT
    if GLOBAL_DASK_CLIENT is not None:
        print("Shutting down Dask client.")
        GLOBAL_DASK_CLIENT.close()
        GLOBAL_DASK_CLIENT = None