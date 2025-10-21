This source code is copied from https://github.com/apache/kyuubi.git,
commit ID 3b205a3924e0e3a75c425de1396089729cf22ee5. We did not modify the code.

The pyhive package is marked as not supported, but we need it to work with Spark.
The latest published version of pyhive on pypi.org is not compatible with current versions of
sqlalchemy and Apache Spark. Specifically, we require the patch made in commit ID
a0b9873f817267675eab304f6935bafa4ab0f731.

We have validated this version of pyhive with our use cases. We will remove this code as
soon as Kyuubi publishes an updated version on pypi.org.

The pyhive license file is included here.
