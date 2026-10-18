from pathlib import Path
import os
from orchestrator.state_store import StateStore
from orchestrator.reporting import write_campaign_report

if __name__ == '__main__':
    runtime = Path(os.getenv('LOCO_RUNTIME','.runtime'))
    print(write_campaign_report(StateStore(), runtime/'exports'))
