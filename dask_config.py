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

def get_or_create_dask_client(config: DaskConfig = None) -> Client:
    global GLOBAL_DASK_CLIENT
    if config is None:
        config = DaskConfig()

    if GLOBAL_DASK_CLIENT is None:
        if config.cluster_type == "local":
            cluster = LocalCluster(
                n_workers=sum(config.n_workers),
                threads_per_worker=1,
                processes=False,  # <-- Critical: Use threads, not processes
                dashboard_address=f":{config.dashboard_port}",
            )
            GLOBAL_DASK_CLIENT = Client(cluster)
    
    return GLOBAL_DASK_CLIENT

def close_dask_client():
    """Close the global Dask client if it exists."""
    global GLOBAL_DASK_CLIENT
    if GLOBAL_DASK_CLIENT is not None:
        print("Shutting down Dask client.")
        GLOBAL_DASK_CLIENT.close()
        GLOBAL_DASK_CLIENT = None