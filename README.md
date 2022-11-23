# Csv To Tfrecords
A small python library to convert a csv file into a tfrecords file

## How to use

```python
import requests
from CsvToTfrecords import c2t

url = 'https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv?raw=true'
csv = requests.get(url)
open('data.csv', 'wb').write(csv.content)

config = {
    'header': [
        'pickup_community_area',
        'fare',
        'trip_start_month',
        'trip_start_hour',
        'trip_start_day',
        'trip_start_timestamp',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude',
        'trip_miles',
        'pickup_census_tract',
        'dropoff_census_tract',
        'payment_type',
        'company',
        'trip_seconds',
        'dropoff_community_area',
        'tips',
    ],
    'integers': [
        'pickup_community_area',
        'trip_start_month',
        'trip_start_hour',
        'trip_start_day',
        'trip_start_timestamp',
        'pickup_census_tract',
        'dropoff_census_tract',
        'trip_seconds',
        'dropoff_community_area',
    ],
    'floats': [
        'fare',
        'pickup_latitude',
        'dropoff_latitude',
        'dropoff_longitude',
        'trip_miles',
        'tips',
    ],
    'categoricals': [
        'payment_type',
        'company',
    ],
}

c2t('./data.csv', './dataset/tfrecords/data.tfrecords', config)
```
