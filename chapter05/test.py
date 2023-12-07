def add_positive_numbers(num1, num2):
    assert num1 > 0, "Both numbers must be positive"
    assert num2 > 0, "Both numbers must be positive"
    return num1 + num2

print(add_positive_numbers(1, 2))  # This will return 3

print(add_positive_numbers(-1, 2))  # This will raise an AssertionError