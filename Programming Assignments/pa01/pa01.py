def world_greeter():
    # This function returns the string "Hello World!"
    return "Hello World!"


def add_a_b(a,b):
    # This function computes and returns the sum of the input variables a and b
    sum_a_b = 0
    sum_a_b = a + b
    return sum_a_b


def smallest_element(list_of_numbers):
    # This function finds and returns the smallest element out of a given list of numerical numbers.
    smallest_number = 0
    smallest_number = min(list_of_numbers)
    return smallest_number


def main():
    # DSL PA01
    print(world_greeter())


if __name__ == "__main__":
    main()
