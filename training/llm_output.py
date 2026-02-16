class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            raise IndexError("Cannot pop from an empty stack")

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            raise IndexError("Cannot peek into an empty stack")

    def is_empty(self):
        return len(self.stack) == 0

# Example usage:
stack = Stack()
stack.push(1)
stack.push(2)
print(stack.peek())  # prints: 2
print(stack.pop())   # prints: 2
print(stack.is_empty())  # prints: False