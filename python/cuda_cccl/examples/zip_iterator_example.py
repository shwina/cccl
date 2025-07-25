"""
Example demonstrating ZipIterator usage in CCCL.

This example shows how to use ZipIterator to combine multiple iterators
for use in parallel algorithms.
"""

import numpy as np
import cuda.cccl.parallel.experimental as cccl
from cuda.cccl.parallel.experimental.iterators import ZipIterator, CountingIterator


def example_zip_with_counting_iterators():
    """Example: Zip two counting iterators together."""
    print("=== Zip with Counting Iterators ===")
    
    # Create counting iterators
    indices = CountingIterator(0)      # [0, 1, 2, 3, 4, ...]
    values = CountingIterator(100)     # [100, 101, 102, 103, 104, ...]
    
    # Create zip iterator
    zip_it = ZipIterator(indices, values)
    
    print(f"Zip iterator created with value type: {zip_it.value_type}")
    print(f"Zip iterator kind: {zip_it.kind}")
    
    # In practice, you would use this in algorithms like:
    # result = cccl.transform(zip_it, zip_it + 5, output_it, some_function)


def example_zip_with_device_arrays():
    """Example: Zip device arrays together."""
    print("\n=== Zip with Device Arrays ===")
    
    # Create device arrays
    keys = np.array([1, 2, 1, 3, 2, 1], dtype=np.int32)
    values = np.array([10.5, 20.5, 30.5, 40.5, 50.5, 60.5], dtype=np.float32)
    
    # Create zip iterator
    zip_it = ZipIterator(keys, values)
    
    print(f"Zip iterator created with value type: {zip_it.value_type}")
    print(f"Input arrays: keys={keys}, values={values}")
    
    # This could be used in reduce_by_key:
    # result = cccl.reduce_by_key(zip_it, zip_it + len(keys), ...)


def example_zip_mixed_types():
    """Example: Zip different types of iterators together."""
    print("\n=== Zip Mixed Iterator Types ===")
    
    # Create different types of iterators
    counting_it = CountingIterator(0)
    constant_it = cccl.ConstantIterator(42)
    
    # Create zip iterator
    zip_it = ZipIterator(counting_it, constant_it)
    
    print(f"Zip iterator created with value type: {zip_it.value_type}")
    
    # This would yield tuples like (0, 42), (1, 42), (2, 42), ...


def example_zip_three_iterators():
    """Example: Zip three iterators together."""
    print("\n=== Zip Three Iterators ===")
    
    # Create three different iterators
    indices = CountingIterator(0)
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    flags = cccl.ConstantIterator(True)
    
    # Create zip iterator
    zip_it = ZipIterator(indices, values, flags)
    
    print(f"Zip iterator created with value type: {zip_it.value_type}")
    
    # This would yield tuples like (0, 1.1, True), (1, 2.2, True), ...


def example_zip_in_transform():
    """Example: Using zip iterator in a transform operation."""
    print("\n=== Zip in Transform Operation ===")
    
    # Create input data
    x_coords = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y_coords = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    
    # Create zip iterator
    coords_it = ZipIterator(x_coords, y_coords)
    
    # Define a function to compute distance from origin
    def distance_from_origin(coord_tuple):
        x, y = coord_tuple
        return np.sqrt(x*x + y*y)
    
    # Create transform iterator (conceptual - would need proper implementation)
    # distance_it = cccl.TransformIterator(coords_it, distance_from_origin)
    
    print(f"Zip iterator for coordinates created")
    print(f"Input: x={x_coords}, y={y_coords}")
    print("Would compute distances from origin for each coordinate pair")


def example_zip_for_reduce_by_key():
    """Example: Using zip iterator for reduce_by_key operations."""
    print("\n=== Zip for Reduce by Key ===")
    
    # Create key-value pairs
    keys = np.array([1, 2, 1, 3, 2, 1, 3], dtype=np.int32)
    values = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32)
    
    # Create zip iterator
    kv_pairs = ZipIterator(keys, values)
    
    print(f"Zip iterator for key-value pairs created")
    print(f"Keys: {keys}")
    print(f"Values: {values}")
    
    # This could be used in reduce_by_key to sum values for each key:
    # result_keys, result_values = cccl.reduce_by_key(
    #     kv_pairs, kv_pairs + len(keys), 
    #     output_keys_it, output_values_it, 
    #     cccl.plus, cccl.plus
    # )


def main():
    """Run all zip iterator examples."""
    print("ZipIterator Examples for CCCL")
    print("=" * 40)
    
    example_zip_with_counting_iterators()
    example_zip_with_device_arrays()
    example_zip_mixed_types()
    example_zip_three_iterators()
    example_zip_in_transform()
    example_zip_for_reduce_by_key()
    
    print("\n" + "=" * 40)
    print("All examples completed!")
    print("\nNote: These examples demonstrate the ZipIterator API.")
    print("Actual algorithm usage would require integration with CCCL algorithms.")


if __name__ == "__main__":
    main() 