
# Script for restoring the snapshot
curl --header "Content-Type: application/json" \
  --request PUT \
  --data '{"type": "fs","settings": {"location": "'"$(pwd)"'/../Snapshot"}}' \
  http://localhost:9200/_snapshot/my_backup

  
curl -X POST http://localhost:9200/_snapshot/my_backup/snapshot_1/_restore