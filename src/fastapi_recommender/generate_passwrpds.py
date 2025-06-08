import csv
import random
import string
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from models import User  # Adjust this import based on your structure

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_random_password(length=12):
    chars = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return ''.join(random.choice(chars) for _ in range(length))

def create_users_from_csv(csv_path: str, db: Session, output_passwords_file="generated_passwords.csv"):
    raw_passwords = []
    user_count = 0
    max_users = 100

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if user_count >= max_users:
                break

            user_id = row['id_user']
            username = row['username']
            try:
                num_helpful_votes_user = int(row['num_helpful_votes_user']) if row['num_helpful_votes_user'] else None
            except ValueError:
                num_helpful_votes_user = None
            try:
                num_cities = float(row['num_cities']) if row['num_cities'] else None
            except ValueError:
                num_cities = None
            try:
                num_reviews_profile = int(row['num_reviews_profile']) if row['num_reviews_profile'] else None
            except ValueError:
                num_reviews_profile = None
            location_user = row['location_user']

            raw_password = generate_random_password()
            hashed_password = pwd_context.hash(raw_password)

            user = User(
                user_id=user_id,
                username=username,
                hashed_password=hashed_password,
                mode="user" if row.get('mode') == 'user' else "collab"  # Default to "hotel" if not specified
            )
            db.add(user)

            raw_passwords.append({
                "user_id": user_id,
                "username": username,
                "raw_password": raw_password
            })

            user_count += 1

    db.commit()

    with open(output_passwords_file, "w", newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["user_id", "username", "raw_password"])
        writer.writeheader()
        writer.writerows(raw_passwords)

    print(f"Created {user_count} users. Raw passwords saved to {output_passwords_file}")
