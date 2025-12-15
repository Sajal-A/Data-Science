## FastAPI Essential

- **What is FastAPI built on?**
  - FastAPI is built on two key components:
    - Starlette: Handles web requests and async event loops
    - Pydantic: Validates and parse your data
  - This combination gives you:
    - Super fast I/O operations
    - Strict type checking automatically
    - Built-in data validation
  - It's a well-designed combination of proven tools.

- **What is the difference between async def vs def ?**
  - Async is the core principle for performance in FastAPI.
  - Use async def when your endpoint does:
    - Database calls
    - External API requests
    - File read/write operaitons
    - Any I/O bound task.
  - Regular def blocks the entire app while waiting.
  - Async def lets the server handle other requests while waiting.
  - That's where you get real performance gains.

- **How do you handle DB connections?**
  - Dependency injection is the secret sauce in FastAPI
  - With Depends, every request automaticallys gets:
    - a fresh database session
    - proper connection management
    - clean separation of concerns
  - You can use the same pattern for :
    - Authentication checks
    - Configuration loading
    - Cache client injection
  - The benefit: Your endpoint stay clean, testable, and decoupled.
  - Show you understand the pattern, not just the syntax.

- **How does FastAPI validate inputs?**
  - Pydantic handle validation automatically.
  - You define a BaseModel with your expected fields FastAPI then:
    - Auto-validates incoming requests.
    - Auto-serializes response
    - Returns clear error messages if validation fails.
  - This means:
  - No more mannual checks like "if 'name' not in payload":
  - Safer APIs with guaranteed data types
  - Cleaner, more maintainable advantage
  - *automatic validation is a production advantage*

- **What about API documentation?**
  - FastAPI auto-generates interactive docs at /docs.
  - It uses your Python type hints to create:
    - Swagger UI automatically
    - Always in-sync documentation
    - Interactive testing interface
  - Why this matters:
    - Frontend teams can test endpoints without asking you
    - New developers understand APIs immediately
    - Your documentation never goes stale
  - *This saves hours of mannual documentation work and keep contracts in sync.*

- **How do you handle slow operations?**
  - Background tasks keep your API responsive
  - Use BackgroundTasks to offload:
      - Email sending
      - log processing
      - Model retraining triggers
      - Heavy computations
  - The flow:
  - 1. Return responses to users immediately
    2. Process slow tasks in background
    3. User doesn't wait.
