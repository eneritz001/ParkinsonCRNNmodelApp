import os
import json
import hashlib

USERS_DIR = "users"
if not os.path.exists(USERS_DIR):
    os.makedirs(USERS_DIR)

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(name, email, password):
    user_id = email.replace("@", "_").replace(".", "_")
    user_path = os.path.join(USERS_DIR, f"{user_id}.json")
    
    if os.path.exists(user_path):
        return False, "The user already exists."
    
    user_data = {
        "name": name,
        "email": email,
        "password": hash_password(password),
        "history": []
    }
    
    with open(user_path, "w") as f:
        json.dump(user_data, f)
    return True, "Registration successful."

def login_user(email, password):
    user_id = email.replace("@", "_").replace(".", "_")
    user_path = os.path.join(USERS_DIR, f"{user_id}.json")
    
    if not os.path.exists(user_path):
        return None, "User not found."
    
    with open(user_path, "r") as f:
        user_data = json.load(f)
    
    if user_data["password"] == hash_password(password):
        return user_data, "Login successful."
    return None, "Incorrect password."

def save_test_result(email, prob_vocal, prob_lectura):
    user_id = email.replace("@", "_").replace(".", "_")
    user_path = os.path.join(USERS_DIR, f"{user_id}.json")
    
    with open(user_path, "r") as f:
        user_data = json.load(f)
    
    import datetime
    new_entry = {
        "date": datetime.datetime.now().strftime("%d/%m/%Y"),
        "vocal": round(prob_vocal * 100, 1),
        "lectura": round(prob_lectura * 100, 1),
        "average": round(((prob_vocal + prob_lectura) / 2) * 100, 1)
    }
    user_data["history"].append(new_entry)
    
    with open(user_path, "w") as f:
        json.dump(user_data, f)
    return user_data