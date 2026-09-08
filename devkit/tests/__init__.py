"""The devkit suite.

A package, unlike the repository's other test directories, because these modules
share a namespace with them under pytest's rootless import mode: this tree's
`test_arrays.py` and `python/marrow/tests/test_arrays.py` would otherwise
collide on a bare module name and abort collection.  Being inside a real
package makes them `devkit.tests.*` and immune to it.
"""
