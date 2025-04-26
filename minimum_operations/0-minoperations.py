def minOperations(n):
    """
    Calculates the minimum number of operations needed to get exactly n 'H' characters.

    Operations allowed:
    - Copy All
    - Paste

    Args:
        n (int): The target number of 'H' characters.

    Returns:
        int: The minimum number of operations, or 0 if impossible.
    """
    if n <= 1:
        # If n is 0 or 1, it's impossible or already done (no operations needed)
        return 0

    operations = 0  # Counter for number of operations
    divisor = 2     # Start checking divisibility from 2

    # Loop until we reduce n down to 1
    while n > 1:
        # While n is divisible by the current divisor
        while n % divisor == 0:
            # Add the divisor to the operation count
            operations += divisor
            # Divide n by the divisor
            n //= divisor
        # Increment the divisor to check next possible factor
        divisor += 1

    return operations
