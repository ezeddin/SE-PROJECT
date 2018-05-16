#!/bin/bash

# Script for downloading and unpacking Elasticsearch 
cd ..
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.2.4.tar.gz
tar -xvzf elasticsearch-6.2.4.tar.gz
rm elasticsearch-6.2.4.tar.gz
echo 'path.repo: ["'"$(pwd)"'/Snapshot"]' >> $(pwd)/elasticsearch-6.2.4/config/elasticsearch.yml
cd mac_scripts