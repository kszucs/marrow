fn main():
    # Test the problematic expression
    var bit_end = 13  # Example: bit position 13, so (13-1) & 7 = 12 & 7 = 4
    
    # Current (buggy) code: ((bit_end - 1) & 7) + 1
    # Due to precedence: (bit_end - 1) & (7 + 1) = 12 & 8 = 8
    var result_buggy = ((bit_end - 1) & 7) + 1
    print("Buggy precedence result:", result_buggy)
    
    # Intended code: (((bit_end - 1) & 7) + 1)
    var result_correct = (((bit_end - 1) & 7) + 1)
    print("Correct parentheses result:", result_correct)
    
    # Now check the shift amounts
    var shift_buggy = 1 << (((bit_end - 1) & 7) + 1)
    print("1 << buggy shift amount:", shift_buggy)
    
    var shift_correct = 1 << ((((bit_end - 1) & 7) + 1))
    print("1 << correct shift amount:", shift_correct)
