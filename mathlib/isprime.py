from number_theory import is_prime

if __name__ == "__main__":
    import sys
    args = sys.argv
    n = int(args[-1])
    if is_prime(n):
      print("True")
    else:
      print("False")