Essential Python Interview Questions:

Core Python
1. What is the difference between list, tuple, set, dictionary?
	- In Python, list, tuple, set, and dictionary are all built-in collection data types, but they differ in mutability, ordering, uniqueness, and access patterns.
	- List(list): A list is an ordered, mutable collection that allows duplicate elements.
		- When to use: 
			- When order matters
			- When elements need to change
			- When you need index-based access
	- Tuple(tuple): A tuple is an ordered, immutable collection that allows duplicate elements.
		- Why use tuple over list?
			- Protects data from accidental modification
			- Slightly more memory-efficient 
			- Suitable for fixed data
		- When to use
			- When data should not change
			- as keys in dictionaries
			- to represent records(e.g., (id, name, age))
	- Set(set): A set is an unordered, mutable collection of unique elements.
		- Common use case:
			- Removing duplicates
			- Membership testing
			- Mathematical operations (union, intersection)
		- Example:
			- A = {1, 2, 3}
			- B = {3, 4, 5}
			- print(A & B)  # Intersection # {3}
			- print(A | B)  # Union # {1, 2, 3, 4, 5}
	- Dictionary(dict) : A dictionary stores data as key-value pairs, where keys are unique and immutable.
		- Charateristics:
			- Ordered (~Python 3.7+)
			- Mutable
			- Fast lookup by key (O(1))
			- Keys must be hashable
		- When to use:
			- When data has a logical mapping.
			- for configuration, JSON-like data.
			- When fast lookups are required.
			
2. What are mutable vs immutable types?
	- Mutable: Objects that can be changed after creation (e.g., list, dict, set).
	- Immutable: Objects that cannot be changed. Any "change" creates a new object in memory (e.g., int, float, str, tuple, frozenset).
	
3. What is __init__ in python classes?
	- It is the constructor method in Python classes. It is automatically called when a new instance (object) of a class is created. It initializes the object's attributes.
	- class Car:
	      def __init__(self, brand):
		      self.brand = brand  # Sets the initial state
4. What are decorators and when do you use them?
	- Decorators are functions that wrap another function to extend its behavior without permanently modifying it. They are commonly used for logging, authentication, and timing.
	def my_decorator(func):
		def wrapper():
			print("Something before.")
			func()
			print("Something after.")
		return wrapper

	@my_decorator
	def say_hello():
		print("Hello!")
		
5. Explain shallow copy vs deep copy.
- Shallow Copy: Creates a new object, but inserts references to the items found in the original. Changes to nested mutable objects affect both.
-Deep Copy: Creates a new object and recursively adds copies of the items found in the original. They are entirely independent.

6. What is list comprehension?
- List comprehension is a concise and expressive way to create lists in Python using a single line of code.
- It allows you to generate a new list by iterating over an iterable, optionally applying a condition, and transforming elements.
	- [expression for item in iterable if condition]
	- For example: squares = [i * i for i in range(5)]
- Why is List Comprehension useful?
	- Concise and Readable: List comprehension improves readability when the logic is simple and linear.
	- Better Performance: Faster than traditional loops due to optimized C-level execution
	
7. What is PEP8 and why does it matter?
- PEP 8 is the Python Enhancement Proposal that provides guidelines on how to write Python code (style guide). It matters because it ensures consistency, making code easier for teams to read and maintain.

8. Difference between append() and extend().
- append(item): Adds the passed object as a single element at the end of the list.
-extend(iterable): Iterates over the passed object and adds each element to the list, effectively merging them.

9. How does python handle memory management?
- Python manages memory via:
	- Private Heap Space: Where all objects and data structures reside.
	- Reference Counting: Tracking how many references point to an object.
	- Garbage Collector: Handles "cyclic references" that reference counting misses.
	- pymalloc for efficient small-object allocation.
10. What is __str__ vs __repr__ ?
- __str__: Aimed at the user. It provides a readable/informal string representation.
- __repr__: Aimed at developers. It provides an unambiguous representation used for debugging (ideally, it should look like the code used to create the object).


Internals & Performance
11. What is GIL (Global Interpreter Lock)?
- The GIL is a mutex that allows only one thread to execute Python bytecode at a time. This simplifies memory management but prevents multi-core CPU usage in CPU-bound multi-threaded programs.
12. What is the difference between CPython vs PyPy?
- CPython: The standard, default implementation written in C. It’s highly compatible but slower for long-running loops.
- PyPy: Uses JIT (Just-In-Time) compilation. It is often much faster than CPython for heavy computation but may consume more RAM.

13. When to use generator vs list?
- List: Stores all elements in memory immediately. Best for small datasets or when you need to access elements multiple times.
- Generator: Yields items one at a time (lazy evaluation). Uses significantly less memory for large datasets.

14. How do yield and generators work?
- The yield keyword pauses the function and saves its state, returning a value to the caller. When the generator is called again, it resumes from where it left off.

15. What is MRO(Method Resolution Order)?
- MRO defines the order in which Python looks for a method in a hierarchy of classes. Python uses the C3 Linearization algorithm. You can view it using ClassName.mro().

16. What is grabage collection in python?
- Python primarily uses Reference Counting. When an object's count hits zero, it's deleted. For "Reference Cycles" (A points to B, B points to A), Python uses a Generational Garbage Collector that periodically scans for unreachable cycles.

17. Python threads vs processes?
- Threads: Share the same memory space. Good for I/O-bound tasks (network calls). Limited by the GIL for CPU tasks.
- Processes: Have separate memory spaces. Good for CPU-bound tasks (data processing). Bypasses the GIL but has higher overhead.

18. What is recursion? Pros & Cons?
- Pros: Leads to clean, elegant code for tree traversals or mathematical sequences (like Factorials).
- Cons: Can lead to RecursionError (Stack Overflow) if too deep. Python has a default limit (usually 1000). It is often less memory-efficient than iteration.

OOP/Advanced Concept
19. What is abstraction, encapsulation, inheritance, polymorphism?
- Abstraction: Hiding complex implementation details (using abc module).
- Encapsulation: Restricting access to data (using _ or __ prefixes).
- Inheritance: Allowing a class to derive properties from another.
- Polymorphism: Using a single interface for different data types (e.g., len() works on strings and lists).

20. Difference between classmethod, staticmethod, instancemethod?
- Instancemethod: An instance method is the default method type. It is bound to an object and receives the instance as its first argument (self).
- Classmethod: Takes (cls) as first argument. Can access/modify class state. Used as factory methods. A class method is bound to the class, not the instance.
- Staticmethod: A static method is not bound to either the instance or the class. It behaves like a normal function, but logically belongs to the class.

21. What is multiple inheritance?
-  A feature where a class can inherit attributes and methods from more than one parent class. Python handles this using the MRO (Method Resolution Order) to avoid the "Diamond Problem."

22. What are magic methods (dunder Methods)?
- Special methods surrounded by double underscores (e.g., __add__, __len__). They allow you to define how objects behave with built-in Python operations (like + or len()).

23. What is dataclass and where do you use it?
- A data class is a Python feature that automatically generates boilerplate methods for classes primarily used to store data, making code cleaner, safer, and more maintainable.
- It automatically generates common methods like __init__, __repr__, __eq__, and others based on class attributes.
	from dataclasses import dataclass

	@dataclass
	class Employee:
		name: str
		salary: float
	
	e1 = Employee("Alex", 50000)
	e2 = Employee("Alex", 50000)
	
	print(e1)          # Employee(name='Alex', salary=50000)
	print(e1 == e2)    # True
- Use data classes when:
	- The class mainly stores data
	- You want clean and readable models
	- You need comparison or representation
	
Async / web
24. What is async and await?
- async:
	- async is used to define a coroutine function.
	- Calling it does not execute immediately.
	- It returns a coroutine object.
- await:
	- await is used inside async functions.
	- It pauses the coroutine until the awaited task completes.
	- While waiting, the event loop runs other tasks.
- These keywords are used for Asynchronous Programming. async def defines a coroutine, and await pauses the execution of the coroutine until the awaited task is finished, allowing other tasks to run in the meantime.

25. Difference between thread, coroutine, & process.
- Process: Independent execution, own memory (Parallelism).
- Thread: Shared memory, managed by OS (Concurrency, I/O).
- Coroutine: Cooperative multitasking, managed by the application (Lightweight concurrency, asyncio).

26. FastAPI vs Django - When to use which?
- FastAPI: Lightweight, high performance (asynchronous), uses type hints. Great for Microservices and APIs.
- Django: "Batteries included." Provides ORM, Admin UI, and Auth out of the box. Best for complex, data-driven web applications.

27. What is request-response lifecycle in Django or FastAPI?
- Request: Client sends HTTP request.
- Middleware: Processes request (auth, logging).
- Routing: URL is matched to a specific View/Endpoint.
- Logic: View interacts with Database (ORM) or Service.
- Response: View returns HTTP response (JSON/HTML).

Real-World Python
28. Why do we use virtual Environment?
- They create isolated environments for different projects. This prevents dependency conflicts (e.g., Project A needing Django 3.0 and Project B needing Django 4.0).

29. How does pip install packages internally?
- pip looks up the package on PyPI (Python Package Index).
- It resolves dependencies to ensure no version conflicts.
- It downloads the Wheel (.whl) or Source Distribution.
- It installs the files into the site-packages directory of your Python environment.

30. What is logging? Why does every project need it? 
- Logging records events that happen while a program runs. Unlike print(), logging allows you to:
	- Categorize messages by severity (INFO, ERROR, CRITICAL).
	- Direct output to files, sockets, or external services.
	- Keep a persistent record for debugging production issues.
