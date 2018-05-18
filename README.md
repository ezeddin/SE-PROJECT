# News Recommendation using Topic Modeling

## Elasticsearch (Restore the index)

### Ubuntu

To restore the index from the provided snapshot, run ```bash restore_snapshot.sh```
To start the search engine backend, run ```bash start.sh```

### MacOS
For instructions for MacOS, check the ```README.md``` in the mac_scripts folder.

## LDA / LSA (Topic Modeling)

Install these python libraries from pip:

```numpy```
```sklearn```
```scipy```
```elasticsearch```

Run ```python ./LDA/distance_matrix.py --model LDA ``` for LDA

Run ```python ./LDA/distance_matrix.py --model LSA ``` for LSA
