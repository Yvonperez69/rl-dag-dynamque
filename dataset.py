# tasks_dataset.py

# -------------------------
# EASY
# -------------------------

tasks_easy = [
    "Write a Python function that computes the nth Fibonacci number.",
    "Implement a function that checks if a string is a palindrome.",
    "Write a function that returns the factorial of a number.",
    "Create a function that removes duplicates from a list.",
    "Write a function that counts the number of vowels in a string.",
    "Implement a function that finds the maximum value in a list.",
    "Write a function that sorts a list without using sorted().",
    "Create a function that checks if a number is prime.",
]

# -------------------------
# MEDIUM
# -------------------------

tasks_medium = [
    "Implement quicksort in Python.",
    "Implement binary search on a sorted list.",
    "Write a function that merges two sorted lists.",
    "Implement a stack class with push, pop, and peek methods.",
    "Implement a queue using two stacks.",
    "Create a function that finds the longest substring without repeating characters.",
    "Implement Dijkstra's algorithm for shortest path.",
    "Write a function that detects a cycle in a graph.",
]

# -------------------------
# FILES & PARSING
# -------------------------

tasks_files = [
    "Write a Python function that reads a CSV file and computes column averages.",
    "Create a function that parses a JSON file and extracts specific fields.",
    "Write a script that counts word frequency in a text file.",
    "Implement a function that saves and loads data using pickle.",
]

# -------------------------
# OOP
# -------------------------

tasks_oop = [
    "Design a BankAccount class with deposit and withdraw methods.",
    "Create a simple Library management system using classes.",
    "Implement a basic linked list class.",
    "Design a simple LRU cache class.",
]

# -------------------------
# TESTING & ROBUSTNESS
# -------------------------

tasks_testing = [
    "Write a function and corresponding unit tests using unittest.",
    "Refactor a function to handle edge cases properly.",
    "Implement input validation for a function that processes user data.",
]

# -------------------------
# NUMPY / DATA
# -------------------------

tasks_data = [
    "Write a function that normalizes a NumPy array.",
    "Implement linear regression using NumPy.",
    "Compute the moving average of a time series.",
    "Write a function that computes cosine similarity between vectors.",
]

# -------------------------
# FULL DATASET
# -------------------------

all_tasks = (
    tasks_easy +
    tasks_medium +
    tasks_files +
    tasks_oop +
    tasks_testing +
    tasks_data
)
