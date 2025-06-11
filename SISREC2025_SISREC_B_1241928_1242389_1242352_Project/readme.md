# Project SISREC2025 - FastAPI Recommender System

## Overview

This project implements a hotel recommender system using FastAPI for the backend, SQLAlchemy for database operations, and a hybrid recommendation logic combining collaborative filtering, content-based filtering, and cold-start strategies.

Key features:

- **User Registration and Authentication**: Users can register with profile metadata and login with JWT-based authentication.
- **Hybrid Recommendation**: Combines collaborative filtering and content-based scores, normalized on a 1-10 scale, with city-based penalty adjustments.
- **Cold-Start Handling**: Implements two cold-start modes (user-based and hotel-based) using feature vectors and clustering.
- **Non-Personalized Recommendations**: Provides a generic ranking of hotels based on weighted review scores and hotel class.
- **Frontend**: Static HTML pages for registration, login, and displaying recommendations.

## Project Structure

```
├── frontend/
│   ├── index.html            # Landing page (redirects to login)
│   ├── login.html            # Login form
│   ├── register.html         # Registration form
│   └── recommendations.html  # Recommendations display page
├── generated_passwords.csv   # Dev-only CSV logging user passwords (remove in production)
├── src/
│   └── fastapi_recommender/
│       ├── main.py           # FastAPI application entrypoint
│       ├── database.py       # SQLAlchemy setup and session
│       ├── models.py         # DB models (User, UserModeProfile, HotelModeProfile)
│       ├── auth/
│       │   ├── user_model.py # Pydantic schemas (UserRegister, LoginRequest, Recommendation)
│       │   └── auth_utils.py # JWT token creation and current-user dependency
│       ├── Recommendation_System_Logic_Code/
│       │   ├── recommender_v2.py                # Hybrid recommendation logic
│       │   ├── recommender_v3.py                # Alternative hybrid logic
│       │   ├── recommender_cold_start_def.py    # Cold-start algorithms
│       │   ├── no_multi_criteria_recommender.py # Older recommendation code
│       │   ├── get_non_personalized_recommendations # Non-personalized logic
│       │   ├── supporting files:                  # Data and models
│       │   │   ├── user_hotel_matrix.npz          # User-item interaction matrix
│       │   │   ├── user_similarity_collab.npz     # User similarity matrix
│       │   │   ├── hotel_similarity_matrix.npz    # Hotel similarity matrix
│       │   │   ├── hotel_features.npz             # Hotel content features
│       │   │   ├── user_features_sparse.npz       # User content features
│       │   │   ├── user_features_collab.npz       # User collaborative features
│       │   │   ├── hotel_df.csv                   # Hotel metadata
│       │   │   ├── *.json                          # ID mappings and clusters
│       │   │   ├── *.pkl                          # Scaler and vectorizers
│       │   │   └── *.npy                          # Latent factor matrices
│       ├── seed.py           # Optional script to seed the database
│       └── generate_passwords.py # Dev-only password generation
└── SISREC2025_SISREC_B_1241928_1242389_1242352_Project.pdf  # Project report
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repo-url>
   cd fastapi_recommender
   ```

2. **Create a Python virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:

   - `DATABASE_URL`: SQLAlchemy connection string (e.g., `sqlite:///./test.db`).
   - `SECRET_KEY`: Secret key for JWT tokens.

## Running the Application

Start the FastAPI server:

```bash
uvicorn src.fastapi_recommender.main:app --reload
```

Navigate to [http://localhost:8000](http://localhost:8000), which redirects to the login page.

## API Endpoints

| Method | Path                                | Description                                  |
| ------ | ----------------------------------- | -------------------------------------------- |
| POST   | `/register`                         | Register new user with profile metadata      |
| POST   | `/login`                            | Obtain JWT token                             |
| GET    | `/recommendations/?token=...`       | Personalized recommendations (hybrid + cold) |
| GET    | `/recommendations/non_personalized` | Top hotels without personalization           |

## Frontend

The `frontend/` directory contains simple HTML pages. Customize or replace with your own UI framework.

## Data Management

- **Database**: Use `src/fastapi_recommender/database.py` to configure your DB.
- **Seeding**: Run `python src/fastapi_recommender/seed.py` to populate initial users or hotels.

## Recommendation Logic

- **Hybrid**: `recommender_v3.py` implements user-item collaborative + content-based filtering, normalizing scores 1–10.
- **Cold-Start**: `recommender_cold_start_def.py` handles new users (feature-based similarity) and hotel-mode profiles.
- **Non-Personalized**: Ranks hotels by normalized weighted scores and class.

## Contributing

1. Fork the repo and create a new branch.
2. Make changes and ensure all existing tests pass.
3. Submit a pull request with a clear description.

## License

[MIT License](LICENSE)

