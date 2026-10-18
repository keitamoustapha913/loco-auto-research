import json
from pathlib import Path
from jsonschema import validate

def test_scout_schema_accepts_unverified_evidence_cards():
    schema=json.loads((Path(__file__).parents[1]/'schemas/literature.schema.json').read_text())
    obj={"schema_version":"1.0","research_gap":"logical heatmaps","sources":[{"title":"Paper","url":"https://example.org/paper","method_family":"student teacher","relevance":"normal-only","evidence_summary":"some claim","limitations":"unconfirmed","licensing_notes":"review","verification_status":"UNVERIFIED_AGENT_SUMMARY"}]}
    validate(obj,schema)
