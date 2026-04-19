# EMO-Play

EMO-Play is a backend system designed to support psychological care for children with Autism Spectrum Disorder (ASD) through emotion recognition.

## Features

- Real-time emotion detection using facial analysis
- User management for patients and therapists
- Session-based emotion analysis tracking
- Secure storage of analysis results
- REST API for frontend integration

## Technology Stack

- Python 3.8+
- FastAPI (web framework)
- SQLAlchemy (ORM)
- OpenCV & DeepFace (emotion detection)
- PostgreSQL/SQLite (database)

## Project Structure

The project follows Clean Architecture principles:

```
src/
├── domain/           # Enterprise business rules
│   ├── entities/     # Core business entities
│   ├── repositories/ # Repository interfaces
│   └── services/     # Service interfaces
├── application/      # Application business rules
│   └── use_cases/    # Use case implementations
├── infrastructure/   # Frameworks and tools
│   ├── persistence/ # Database implementation
│   └── ai/          # AI services implementation
└── interfaces/      # Interface adapters
    └── api/         # FastAPI routes and models
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd emo-play
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file:
```bash
cp .env.example .env
```

5. Update the .env file with your configuration.

## Running the Application

1. Start the development server:
```bash
cd src
python main.py
```

2. Access the API documentation at:
```
http://localhost:8000/docs
```

## API Endpoints

### Users
- POST /users/ - Create a new user
- GET /users/{user_id} - Get user details

### Emotion Analysis
- POST /analysis/ - Analyze an image and store results
- GET /analysis/session/{session_id} - Get analysis results for a session
- GET /analysis/user/{user_id} - Get all analysis results for a user

## Development

1. The application uses SQLite for development but can be configured to use PostgreSQL in production.
2. All database models are defined using SQLAlchemy ORM.
3. The FastAPI application provides automatic API documentation via Swagger UI.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request