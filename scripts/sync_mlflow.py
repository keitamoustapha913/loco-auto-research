from orchestrator.state_store import StateStore
from orchestrator.mlflow_tracker import sync_outbox
if __name__ == "__main__":
    print(sync_outbox(StateStore()))
