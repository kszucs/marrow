# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Apache Iceberg tables, read natively.

The table format's own pieces — single-value bounds, partition transforms,
schemas resolved by field id — over marrow's Parquet reader. Table metadata
and manifests (JSON and Avro) build on these.
"""
