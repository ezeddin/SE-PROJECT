#!/bin/bash

BASEPATH=$(pwd)

curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.2.4.tar.gz
tar -xvzf elasticsearch-6.2.4.tar.gz
rm elasticsearch-6.2.4.tar.gz

echo 'path.repo: ["'"$BASEPATH"'/Snapshot"]' >> $BASEPATH/elasticsearch-6.2.4/config/elasticsearch.yml

sleep 1

nohup $BASEPATH/elasticsearch-6.2.4/bin/elasticsearch &>/dev/null &

sleep 15

curl --header "Content-Type: application/json" \
  --request PUT \
  --data '{"type": "fs","settings": {"location": "'"$BASEPATH"'/Snapshot"}}' \
  http://localhost:9200/_snapshot/my_backup

  
curl -X POST http://localhost:9200/_snapshot/my_backup/snapshot_1/_restore

sleep 2

pkill -f Elasticsearch

