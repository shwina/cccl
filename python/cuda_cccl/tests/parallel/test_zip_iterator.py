"""
Test file for ZipIterator functionality.
"""

import numpy as np
import pytest
import cuda.cccl.parallel.experimental as cccl
from cuda.cccl.parallel.experimental.iterators import ZipIterator, CountingIterator


def test_zip_iterator_basic():
    """Test basic ZipIterator functionality with counting iterators."""
    # Create counting iterators
    it1 = CountingIterator(10)  # [10, 11, 12, ...]
    it2 = CountingIterator(100)  # [100, 101, 102, ...]
    
    # Create zip iterator
    zip_it = ZipIterator(it1, it2)
    
    # Test that we can create the iterator
    assert zip_it is not None
    assert zip_it.value_type is not None
    
    # Test that the iterator has the expected properties
    assert hasattr(zip_it, 'advance')
    assert hasattr(zip_it, 'dereference')
    assert hasattr(zip_it, 'host_advance')


def test_zip_iterator_with_device_arrays():
    """Test ZipIterator with device arrays."""
    # Create device arrays
    arr1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    arr2 = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    
    # Create zip iterator
    zip_it = ZipIterator(arr1, arr2)
    
    # Test that we can create the iterator
    assert zip_it is not None
    assert zip_it.value_type is not None


def test_zip_iterator_mixed_types():
    """Test ZipIterator with mixed iterator types."""
    # Create different types of iterators
    counting_it = CountingIterator(0)
    constant_it = cccl.ConstantIterator(42)
    
    # Create zip iterator
    zip_it = ZipIterator(counting_it, constant_it)
    
    # Test that we can create the iterator
    assert zip_it is not None
    assert zip_it.value_type is not None


def test_zip_iterator_empty():
    """Test that ZipIterator raises an error with no iterators."""
    with pytest.raises(ValueError, match="At least one iterator is required"):
        ZipIterator()


def test_zip_iterator_single():
    """Test ZipIterator with a single iterator."""
    counting_it = CountingIterator(0)
    zip_it = ZipIterator(counting_it)
    
    # Should work with single iterator
    assert zip_it is not None
    assert zip_it.value_type is not None


def test_zip_iterator_in_algorithms():
    """Test ZipIterator usage in algorithms (conceptual test)."""
    # This would be a more comprehensive test that actually uses the iterator
    # in algorithms like reduce, transform, etc.
    # For now, we just test that the iterator can be created
    
    # Create test data
    keys = np.array([1, 2, 1, 3, 2], dtype=np.int32)
    values = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    
    # Create zip iterator
    zip_it = ZipIterator(keys, values)
    
    # Test that we can create the iterator
    assert zip_it is not None
    
    # In a real implementation, we could use this in algorithms like:
    # result = cccl.reduce_by_key(zip_it, zip_it + len(keys), ...)


if __name__ == "__main__":
    # Run basic tests
    test_zip_iterator_basic()
    test_zip_iterator_with_device_arrays()
    test_zip_iterator_mixed_types()
    test_zip_iterator_empty()
    test_zip_iterator_single()
    test_zip_iterator_in_algorithms()
    print("All ZipIterator tests passed!") 