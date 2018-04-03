# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Library to upload benchmark generated by BenchmarkLogger to remote repo.

This library require google cloud bigquery lib as dependency, which can be
installed with:
  > pip install --upgrade google-cloud-bigquery
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import uuid

from google.cloud import bigquery

import tensorflow as tf # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers
from official.utils.logging import logger


class BigQueryUploader(object):
  """Upload the benchmark and metric info to BigQuery."""

  def __init__(self, logging_dir, gcp_project=None, credentials=None):
    """Initialized BigQueryUploader with proper setting.

    Args:
      logging_dir: string, logging directory that contains the benchmark log.
      gcp_project: string, the name of the GCP project that the log will be
        uploaded to. The default project name will be detected from local
        environment if no value is provided.
      credentials: google.auth.credentials. The credential to access the
        BigQuery service. The default service account credential will be
        detected from local environment if no value is provided. Please use
        google.oauth2.service_account.Credentials to load credential from local
        file for the case that the test is run out side of GCP.
    """
    self._logging_dir = logging_dir
    self._bq_client = bigquery.Client(
        project=gcp_project, credentials=credentials)

  def upload_benchmark_run(self, dataset_name, table_name, run_id):
    """Upload benchmark run information to Bigquery.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the data will be uploaded.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format.
    """
    expected_file = os.path.join(
        self._logging_dir, logger.BENCHMARK_RUN_LOG_FILE_NAME)
    with tf.gfile.GFile(expected_file) as f:
      benchmark_json = json.load(f)
      benchmark_json["model_id"] = run_id
      table_ref = self._bq_client.dataset(dataset_name).table(table_name)
      errors = self._bq_client.insert_rows_json(table_ref, [benchmark_json])
      if errors:
        tf.logging.error(
            "Failed to upload benchmark info to bigquery: {}".format(errors))

  def upload_metric(self, dataset_name, table_name, run_id):
    """Upload metric information to Bigquery.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the metric data will be uploaded. This is different from the
        benchmark_run table.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format. This should be the same as the benchmark run_id.
    """
    expected_file = os.path.join(
        self._logging_dir, logger.METRIC_LOG_FILE_NAME)
    with tf.gfile.GFile(expected_file) as f:
      lines = f.readlines()
      metrics = []
      for line in filter(lambda l: l.strip(), lines):
        metric = json.loads(line)
        metric["run_id"] = run_id
        metrics.append(metric)
      table_ref = self._bq_client.dataset(dataset_name).table(table_name)
      errors = self._bq_client.insert_rows_json(table_ref, metrics)
      if errors:
        tf.logging.error(
            "Failed to upload benchmark info to bigquery: {}".format(errors))


def main(argv):
  parser = parsers.BenchmarkParser()
  flags = parser.parse_args(args=argv[1:])
  if not flags.benchmark_log_dir:
    print("Usage: benchmark_uploader.py --benchmark_log_dir=/some/dir")
    sys.exit(1)

  uploader = BigQueryUploader(
      flags.benchmark_log_dir,
      gcp_project=flags.gcp_project)
  run_id = str(uuid.uuid4())
  uploader.upload_benchmark_run(
      flags.bigquery_data_set, flags.bigquery_run_table, run_id)
  uploader.upload_metric(
      flags.bigquery_data_set, flags.bigquery_metric_table, run_id)


if __name__ == "__main__":
  main(argv=sys.argv)
